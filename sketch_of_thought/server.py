"""Small dependency-free local web UI for persistent SoT planning chats."""

from __future__ import annotations

import json
import os
import threading
import uuid
from email.parser import BytesParser
from email.policy import default as email_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .chat import (
    PARADIGMS,
    Conversation,
    ConversationStore,
    ProviderError,
    Router,
    adapter_for,
    provider_specs,
    effort_options,
    normalize_effort,
    route_prompt,
    split_thinking,
    utc_now,
)
from .model_catalog import model_catalog


UPLOAD_MAX_FILES = 50
UPLOAD_MAX_FILE_BYTES = 1_000_000
UPLOAD_MAX_TOTAL_BYTES = 6_000_000
UPLOAD_MAX_CONTEXT_CHARS = 120_000
UPLOAD_MAX_REQUEST_BYTES = 8_000_000


def _limit(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _system_prompt(paradigm: str) -> str:
    names = {
        "chunked_symbolism": "EN_ChunkedSymbolism_SystemPrompt.md",
        "conceptual_chaining": "EN_ConceptualChaining_SystemPrompt.md",
        "expert_lexicons": "EN_ExpertLexicons_SystemPrompt.md",
    }
    path = Path(__file__).parent / "config" / "prompts" / "EN" / names[paradigm]
    return path.read_text(encoding="utf-8")


class ChatApplication:
    def __init__(self, store: ConversationStore | None = None) -> None:
        self.store = store or ConversationStore()
        self.router = Router()
        self._conversation_locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    def _lock_for(self, conversation_id: str) -> threading.Lock:
        with self._locks_lock:
            return self._conversation_locks.setdefault(conversation_id, threading.Lock())

    def bootstrap(self) -> dict[str, Any]:
        return {
            "providers": [spec.as_dict() for spec in provider_specs()],
            "models": model_catalog(),
            "paradigms": list(PARADIGMS),
            "efforts": effort_options(),
            "efforts_by_provider": {spec.key: effort_options(spec.key) for spec in provider_specs()},
            "conversations": [self._summary(item) for item in self.store.list()],
        }

    @staticmethod
    def _summary(conversation: Conversation) -> dict[str, Any]:
        return {
            "id": conversation.id,
            "title": conversation.title,
            "updated_at": conversation.updated_at,
            "provider": conversation.provider,
            "model": conversation.model,
            "effort": conversation.effort,
            "plan_mode": conversation.plan_mode,
            "message_count": len(conversation.messages),
            "upload_count": len(conversation.uploads),
        }

    def create(self, payload: dict[str, Any]) -> Conversation:
        conversation = Conversation.new(str(payload.get("title") or "New planning session")[:120])
        provider = str(payload.get("provider") or "claude")
        if provider not in {spec.key for spec in provider_specs()}:
            raise ValueError("unknown provider")
        conversation.provider = provider
        conversation.model = str(payload.get("model") or "")[:200]
        conversation.effort = normalize_effort(payload.get("effort"))
        conversation.plan_mode = _flag(payload.get("plan_mode", True))
        self.store.save(conversation)
        return conversation

    def delete(self, conversation_id: str) -> None:
        self.store.delete(conversation_id)


    @staticmethod
    def _upload_name(value: str) -> str:
        name = Path(str(value or "").replace(chr(92), "/")).name.strip()
        name = "".join(char if 32 <= ord(char) and ord(char) != 127 else "_" for char in name)
        return (name or "uploaded-text")[:200]

    def upload(self, conversation_id: str, files: list[tuple[str, bytes]]) -> Conversation:
        if not files:
            raise ValueError("no files uploaded")
        with self._lock_for(conversation_id):
            conversation = self.store.load(conversation_id)
            max_files = _limit("SOT_UPLOAD_MAX_FILES", UPLOAD_MAX_FILES)
            max_file_bytes = _limit("SOT_UPLOAD_MAX_FILE_BYTES", UPLOAD_MAX_FILE_BYTES)
            max_total_bytes = _limit("SOT_UPLOAD_MAX_TOTAL_BYTES", UPLOAD_MAX_TOTAL_BYTES)
            if len(conversation.uploads) + len(files) > max_files:
                raise ValueError(f"too many uploaded files (maximum {max_files})")

            stored_bytes = 0
            for upload in conversation.uploads:
                try:
                    stored_bytes += int(upload.get("bytes", 0))
                except (TypeError, ValueError):
                    stored_bytes += len(str(upload.get("text") or "").encode("utf-8"))

            additions: list[dict[str, Any]] = []
            for original_name, raw in files:
                raw = bytes(raw)
                if not raw:
                    raise ValueError(f"{self._upload_name(original_name)} is empty")
                if len(raw) > max_file_bytes:
                    raise ValueError(
                        f"{self._upload_name(original_name)} is too large "
                        f"(maximum {max_file_bytes} bytes)"
                    )
                try:
                    text = raw.decode("utf-8-sig")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"{self._upload_name(original_name)} is not valid UTF-8 text"
                    ) from exc
                if chr(0) in text:
                    raise ValueError(f"{self._upload_name(original_name)} does not look like a text file")
                if stored_bytes + sum(item["bytes"] for item in additions) + len(raw) > max_total_bytes:
                    raise ValueError(f"uploaded text exceeds the {max_total_bytes}-byte conversation limit")
                additions.append({
                    "id": str(uuid.uuid4()),
                    "name": self._upload_name(original_name),
                    "text": text,
                    "bytes": len(raw),
                    "chars": len(text),
                    "created_at": utc_now(),
                })

            conversation.uploads.extend(additions)
            conversation.updated_at = utc_now()
            self.store.save(conversation)
            return conversation

    def clear_uploads(self, conversation_id: str) -> Conversation:
        with self._lock_for(conversation_id):
            conversation = self.store.load(conversation_id)
            conversation.uploads = []
            conversation.updated_at = utc_now()
            self.store.save(conversation)
            return conversation


    def send(self, conversation_id: str, payload: dict[str, Any]) -> Conversation:
        text = str(payload.get("message") or "").strip()
        if not text:
            raise ValueError("message is required")

        with self._lock_for(conversation_id):
            conversation = self.store.load(conversation_id)
            provider = str(payload.get("provider") or conversation.provider)
            model = str(payload.get("model") if payload.get("model") is not None else conversation.model).strip()[:200]
            effort = normalize_effort(payload.get("effort") if payload.get("effort") is not None else conversation.effort)
            plan_mode = _flag(payload.get("plan_mode") if payload.get("plan_mode") is not None else conversation.plan_mode)
            if provider not in {spec.key for spec in provider_specs()}:
                raise ValueError("unknown provider")

            selected = str(payload.get("paradigm") or "auto")
            route = self.router.classify(text) if selected == "auto" else {
                "paradigm": selected,
                "confidence": 1.0,
                "backend": "manual",
            }
            paradigm = route["paradigm"]
            if paradigm not in PARADIGMS:
                raise ValueError("unknown paradigm")

            include_uploads = _flag(payload.get("include_uploads"))
            upload_names = [str(upload.get("name") or "uploaded text") for upload in conversation.uploads]
            uploaded_context = conversation.upload_context(
                max_chars=_limit("SOT_UPLOAD_MAX_CONTEXT_CHARS", UPLOAD_MAX_CONTEXT_CHARS)
            ) if include_uploads else ""

            spec_key = f"{provider}:{model}"
            session = conversation.provider_sessions.get(spec_key, {})
            native_session_id = session.get("session_id")
            initial = not bool(native_session_id)
            handoff = None
            if not native_session_id and conversation.messages:
                budget = int(os.environ.get("SOT_HANDOFF_MAX_CHARS", "24000"))
                handoff = conversation.handoff(max_chars=max(4000, budget))

            system_prompt = _system_prompt(paradigm)
            prompt = route_prompt(paradigm, "", text, handoff=handoff)
            if uploaded_context:
                prompt += "\n\n" + uploaded_context

            conversation.messages.append({
                "role": "user",
                "content": text,
                "paradigm": paradigm,
                "router": route,
                "provider": provider,
                "model": model,
                "effort": effort,
                "include_uploads": bool(uploaded_context),
                "uploaded_files": upload_names if uploaded_context else [],
                "created_at": utc_now(),
            })
            conversation.provider = provider
            conversation.model = model
            conversation.effort = effort
            conversation.plan_mode = plan_mode
            conversation.updated_at = utc_now()
            self.store.save(conversation)

            try:
                result = adapter_for(provider).invoke(
                    prompt,
                    model=model,
                    native_session_id=native_session_id,
                    effort=effort,
                    plan_mode=plan_mode,
                    initial=initial,
                    system_prompt=system_prompt,
                    cwd=os.environ.get("SOT_WORKDIR", os.getcwd()),
                )
            except ProviderError:
                # Keep the user message so retrying does not silently lose it.
                raise

            thinking, response = split_thinking(result.text, structured_thinking=result.thinking)
            conversation.messages.append({
                "role": "assistant",
                "content": result.text,
                "response": response,
                "thinking": thinking,
                "paradigm": paradigm,
                "router": route,
                "provider": provider,
                "model": model,
                "effort": effort,
                "include_uploads": bool(uploaded_context),
                "uploaded_files": upload_names if uploaded_context else [],
                "created_at": utc_now(),
            })
            conversation.provider_sessions[spec_key] = {
                "session_id": result.session_id,
                "updated_at": utc_now(),
            }
            conversation.updated_at = utc_now()
            self.store.save(conversation)
            return conversation


INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SoT planning chat</title>
<style>
:root{color-scheme:dark;--bg:#101216;--panel:#171a20;--line:#2b303b;--muted:#9ca5b5;--accent:#8ec5ff;--user:#1c3148;--assistant:#20252d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf1f7;font:14px system-ui,-apple-system,Segoe UI,sans-serif;height:100vh;display:grid;grid-template-columns:260px minmax(0,1fr);min-width:0;min-height:0;overflow-x:hidden}
aside{border-right:1px solid var(--line);padding:16px;overflow:auto;background:#13161b}h1{font-size:16px;margin:0 0 14px}button,select,input,textarea{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:8px}button{cursor:pointer}button:disabled{opacity:.45;cursor:not-allowed}button:hover{border-color:var(--accent)}#new{width:100%;margin-bottom:14px}.conversation{display:block;width:100%;text-align:left;margin:5px 0;background:transparent;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.conversation.active{border-color:var(--accent);background:#1b2837}.small{font-size:12px;color:var(--muted)}main{display:grid;grid-template-rows:auto auto auto minmax(0,1fr) auto;min-width:0;min-height:0}.toolbar{padding:12px 18px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center;flex-wrap:wrap;min-width:0}.toolbar label{color:var(--muted);font-size:12px}.toolbar select{min-width:190px}.toolbar input{width:220px}.toolbar input.upload-input{width:235px;min-width:180px;padding:5px}.include-upload{display:flex;align-items:center;gap:4px;white-space:nowrap}.include-upload input{width:auto;padding:0}.plan-mode-group{display:none;align-items:center;gap:10px;font-size:12px;color:var(--muted);white-space:nowrap}.plan-mode-group.visible{display:flex}.plan-mode-group label{display:flex;align-items:center;gap:3px}.plan-mode-group input{width:auto;padding:0}.upload-status{font-size:12px;color:var(--muted);max-width:28ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.model-select{min-width:280px}.route{margin-left:auto;color:var(--accent);font-size:12px;border:1px solid var(--line);border-radius:999px;padding:5px 9px;background:#151c25;max-width:44ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.view-tabs{display:flex;gap:6px;padding:10px 18px 0}.view-tab{border-radius:999px;padding:5px 10px;color:var(--muted);background:transparent}.view-tab.active{color:#edf1f7;background:#25364a;border-color:#3975ad}.delete-session{width:100%;margin-top:5px;color:#f2a3a3}.chat{width:100%;min-width:0;min-height:0;overflow:auto;overscroll-behavior:contain;padding:24px max(18px,calc((100vw - 920px)/2));display:flex;flex-direction:column;align-items:stretch;gap:14px}.bubble{width:min(860px,100%);max-width:100%;min-width:0;border:1px solid var(--line);border-radius:10px;padding:12px 14px;white-space:pre-wrap;line-height:1.5;overflow:visible;overflow-wrap:anywhere;word-break:break-word}.bubble>div{max-width:100%;min-width:0;overflow:visible;overflow-wrap:anywhere;word-break:break-word}.bubble.user{align-self:flex-end;background:var(--user)}.bubble.assistant{align-self:flex-start;background:var(--assistant)}.meta{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--muted);margin-bottom:6px}.copy-button{margin-left:auto;padding:2px 8px;font-size:11px;line-height:1.2;color:var(--muted);background:transparent}.composer{border-top:1px solid var(--line);padding:14px max(18px,calc((100vw - 920px)/2));display:flex;gap:8px}.composer textarea{resize:vertical;min-height:54px;flex:1}.composer button{align-self:flex-end;background:#254a70;border-color:#3975ad}.empty{color:var(--muted);margin:auto;text-align:center}.warning{color:#f2c879;font-size:12px;margin:0 18px 8px}@media(max-width:700px){body{grid-template-columns:1fr}aside{display:none}.route{width:100%;margin-left:0}.toolbar input{width:150px}}
</style></head>
<body><aside><h1>SoT planning chats</h1><button id="new">＋ New session</button><button id="delete" class="delete-session" disabled>Delete session</button><div id="sessions"></div><p class="small">The transcript is stored locally. Harness sessions stay native where supported; switching harnesses uses a bounded text handoff.</p></aside>
<main><div class="toolbar"><label for="provider">Harness</label><select id="provider"></select><label for="model">Model</label><select id="model" class="model-select"></select><input id="custom-model" placeholder="Custom model ID (optional)"><label for="effort">Effort</label><select id="effort"></select><span class="plan-mode-group" id="plan-mode-group" title="Plan mode: read-only, no tools. Full access: tools enabled, permission prompts bypassed."><label><input type="radio" name="plan-mode" id="plan-mode-on" value="on"> Plan mode</label><label><input type="radio" name="plan-mode" id="plan-mode-off" value="off"> Full access</label></span><label for="upload-files">Context files</label><input id="upload-files" class="upload-input" type="file" multiple accept=".txt,.md,.csv,.json,.yaml,.yml,.xml,.html,.log,.py,.js,.ts,.rst,.tex,text/*"><label class="include-upload"><input id="include-uploads" type="checkbox"> Include uploaded texts in next prompt</label><button id="clear-uploads" type="button">Clear</button><span class="upload-status" id="upload-status">No uploaded texts</span><label for="paradigm">Router</label><select id="paradigm"><option value="auto">Auto</option><option value="conceptual_chaining">Conceptual chaining</option><option value="chunked_symbolism">Chunked symbolism</option><option value="expert_lexicons">Expert lexicons</option></select><span class="route" id="route">No turn yet</span></div> <nav class="view-tabs" aria-label="Chat view"><button class="view-tab active" data-view="response">Response log</button><button class="view-tab" data-view="thinking">Thinking log</button></nav><div class="warning" id="warning"></div><section class="chat" id="chat"><div class="empty">Create a session and start planning.</div></section><form class="composer" id="composer"><textarea id="message" placeholder="Describe what you are planning..."></textarea><button>Send</button></form></main>
<script>
let state={boot:null,conversation:null,view:"response",sending:false};
const byId=id=>document.getElementById(id);
async function api(path,options){let r=await fetch(path,Object.assign({headers:{"content-type":"application/json"}},options||{}));let d=await r.json();if(!r.ok)throw Error(d.error||"Request failed");return d;}
function selectedModel(){let custom=byId("custom-model").value.trim();return custom||byId("model").value;}
function selectedEffort(){return byId("effort").value;}
function selectedPlanMode(){return !byId("plan-mode-off").checked;}
function planModeSupported(providerKey){let spec=(state.boot.providers||[]).find(function(p){return p.key===providerKey;});return !!(spec&&spec.supports_plan_mode);}
function renderPlanMode(){
  let supported=planModeSupported(byId("provider").value);
  byId("plan-mode-group").classList.toggle("visible",supported);
  let current=state.conversation?state.conversation.plan_mode:true;
  if(current===false){byId("plan-mode-off").checked=true;}else{byId("plan-mode-on").checked=true;}
}
function selectedModelEntry(){let provider=byId("provider").value;let model=selectedModel();return ((state.boot&&state.boot.models&&state.boot.models[provider])||[]).find(function(item){return item.id===model;})||null;}function renderEfforts(){let select=byId("effort");select.innerHTML="";let entry=selectedModelEntry();let options;if(entry&&Array.isArray(entry.supported_efforts)&&entry.supported_efforts.length){options=[{id:"",label:"Harness default"}].concat(entry.supported_efforts.map(function(id){return {id:id,label:id==="xhigh"?"Extra high":id.charAt(0).toUpperCase()+id.slice(1)};}));}else{options=(state.boot&&state.boot.efforts_by_provider&&state.boot.efforts_by_provider[byId("provider").value])||(state.boot&&state.boot.efforts)||[{id:"",label:"Harness default"}];}let current=state.conversation?state.conversation.effort:"";options.forEach(function(e){let option=document.createElement("option");option.value=e.id;option.textContent=e.label;select.appendChild(option);});select.value=options.some(function(e){return e.id===current;})?current:"";}
function renderUploads(){
  let uploads=(state.conversation&&state.conversation.uploads)||[];
  let fileInput=byId("upload-files");
  let include=byId("include-uploads");
  let clear=byId("clear-uploads");
  let status=byId("upload-status");
  let names=uploads.map(function(upload){return upload.name||"uploaded text";});
  fileInput.disabled=!state.conversation||state.sending;
  include.disabled=!state.conversation||!uploads.length||state.sending;
  clear.disabled=!state.conversation||!uploads.length||state.sending;
  status.textContent=names.length?(names.length+" file"+(names.length===1?"":"s")+": "+names.join(", ")):"No uploaded texts";
  status.title=names.join(", ");
}
async function uploadFiles(){
  let input=byId("upload-files");
  let files=Array.from(input.files||[]);
  if(!files.length||!state.conversation)return;
  let form=new FormData();
  files.forEach(function(file){form.append("files",file,file.name);});
  try{
    let response=await fetch("/api/conversations/"+encodeURIComponent(state.conversation.id)+"/uploads",{method:"POST",body:form});
    let result=await response.json();
    if(!response.ok)throw Error(result.error||"Upload failed");
    state.conversation=result;
    let summary=state.boot.conversations.find(function(item){return item.id===result.id;});
    if(summary){summary.upload_count=(result.uploads||[]).length;summary.updated_at=result.updated_at;}
    input.value="";
    renderUploads();
    renderSessions();
    byId("warning").textContent="";
  }catch(error){byId("warning").textContent=error.message;input.value="";renderUploads();}
}
async function clearUploads(){
  if(!state.conversation||state.sending)return;
  if(window.confirm("Clear uploaded texts from this session?")===false)return;
  try{
    let result=await api("/api/conversations/"+encodeURIComponent(state.conversation.id)+"/uploads",{method:"DELETE"});
    state.conversation=result;
    byId("include-uploads").checked=false;
    let summary=state.boot.conversations.find(function(item){return item.id===result.id;});
    if(summary){summary.upload_count=0;summary.updated_at=result.updated_at;}
    renderUploads();
    renderSessions();
  }catch(error){byId("warning").textContent=error.message;}
}
function renderModels(){let provider=byId("provider").value;let models=(state.boot.models&&state.boot.models[provider])||[];let current=state.conversation?state.conversation.model:"";let select=byId("model");select.innerHTML="";let defaultOption=document.createElement("option");defaultOption.value="";defaultOption.textContent="Use harness default";select.appendChild(defaultOption);models.forEach(function(m){let option=document.createElement("option");option.value=m.id;option.textContent=m.label+" ("+m.id+")";select.appendChild(option);});if(current&&models.some(function(m){return m.id===current;})){select.value=current;byId("custom-model").value="";}else{select.value="";byId("custom-model").value=current||"";}}
function renderProviders(){byId("provider").innerHTML="";state.boot.providers.forEach(function(p){let option=document.createElement("option");option.value=p.key;option.disabled=!p.available;option.textContent=p.label+(p.available?"":" (not installed)");byId("provider").appendChild(option);});byId("paradigm").value="auto";renderModels();renderEfforts();renderPlanMode();}
function prettyParadigm(value){return ({conceptual_chaining:"Conceptual chaining",chunked_symbolism:"Chunked symbolism",expert_lexicons:"Expert lexicons"}[value]||value||"Unknown");}
function splitThinking(content, thinkingHint, responseHint){
  let source=String(content==null?"":content);
  let response=source;
  let thinking=[];
  let patterns=[
    /<\s*(think|thinking|analysis|reasoning)\s*>([\s\S]*?)<\s*\/\s*(think|thinking|analysis|reasoning)\s*>/gi,
    /<\|\s*(think|thinking|analysis|reasoning)\s*\|>([\s\S]*?)<\|\s*\/\s*(think|thinking|analysis|reasoning)\s*\|>/gi,
    /\[\s*(think|thinking|analysis|reasoning)\s*\]([\s\S]*?)\[\s*\/\s*(think|thinking|analysis|reasoning)\s*\]/gi
  ];
  patterns.forEach(function(pattern){response=response.replace(pattern,function(_,name,body){if(body.trim())thinking.push(body.trim());return "";});});
  let opening=/<\s*(think|thinking|analysis|reasoning)\s*>|<\|\s*(think|thinking|analysis|reasoning)\s*\|>|\[\s*(think|thinking|analysis|reasoning)\s*\]/i.exec(response);
  if(opening){
    let answer=/\\boxed\s*\{/i.exec(response.slice(opening.index+opening[0].length));
    if(answer){let answerStart=opening.index+opening[0].length+answer.index;if(response.slice(opening.index+opening[0].length,answerStart).trim())thinking.push(response.slice(opening.index+opening[0].length,answerStart).trim());response=response.slice(0,opening.index)+response.slice(answerStart);}
    else{if(response.slice(opening.index+opening[0].length).trim())thinking.push(response.slice(opening.index+opening[0].length).trim());response=response.slice(0,opening.index);}
  }
  if(thinkingHint&&String(thinkingHint).trim())thinking.unshift(String(thinkingHint).trim());
  response=responseHint==null?response:String(responseHint);
  return {thinking:[...new Set(thinking)].join(String.fromCharCode(10)+String.fromCharCode(10)).trim(),response:response.trim()};
}

function setView(view){state.view=view;document.querySelectorAll(".view-tab").forEach(function(tab){tab.classList.toggle("active",tab.dataset.view===view);});renderChat();}
function renderRouter(){let c=state.conversation;let assistant=c&&c.messages.slice().reverse().find(function(m){return m.role==="assistant";});let route=assistant&&assistant.router;if(!route&&c){let users=c.messages.filter(function(m){return m.role==="user";});route=users.length?users[users.length-1].router:null;}let routeElement=byId("route");if(!assistant||!route){routeElement.textContent="No turn yet";routeElement.title="The selected SoT paradigm will appear here.";return;}let confidence=route.confidence==null?"not available":String(route.confidence);let label=prettyParadigm(assistant.paradigm||route.paradigm);let detail="Last: "+label+" | "+(route.backend||"unknown")+" | confidence "+confidence+" | "+(assistant.provider||"unknown")+" / "+(assistant.model||"harness default");routeElement.textContent=detail;routeElement.title=detail;}
function updateDeleteButton(){byId("delete").disabled=!state.conversation;}
function renderSessions(){let container=byId("sessions");container.innerHTML="";(state.boot.conversations||[]).forEach(function(c){let button=document.createElement("button");button.className="conversation "+(state.conversation&&state.conversation.id===c.id?"active":"");button.textContent=c.title;button.dataset.id=c.id;button.onclick=function(){openConversation(c.id);};container.appendChild(button);});updateDeleteButton();}
function readLatexGroup(source,start){if(source[start]!=="{")return null;let depth=0;for(let i=start;i<source.length;i++){if(source[i]==="{")depth++;else if(source[i]==="}"){depth--;if(depth===0)return {value:source.slice(start+1,i),end:i+1};}}return null;}
function unwrapLatexCommand(source,command){let marker="\\"+command;let cursor=0;let result="";while(cursor<source.length){let index=source.indexOf(marker,cursor);if(index<0){result+=source.slice(cursor);break;}let brace=source.indexOf("{",index+marker.length);let group=brace<0?null:readLatexGroup(source,brace);if(!group){result+=source.slice(cursor,index)+source.slice(index+marker.length);cursor=index+marker.length;continue;}result+=source.slice(cursor,index)+group.value;cursor=group.end;}return result;}
function cleanLatex(value){let result=String(value==null?"":value);["boxed","text","mathrm","operatorname","mathbf","mathit","underline"].forEach(function(command){for(let i=0;i<5;i++){let next=unwrapLatexCommand(result,command);if(next===result)break;result=next;}});result=result.replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g,"$1/$2").replace(/\\sqrt\s*\{([^{}]*)\}/g,"√$1");let symbols={times:"×",cdot:"·",rightarrow:"→",to:"→",leq:"≤",geq:"≥",pm:"±",neq:"≠",approx:"≈",degree:"°"};result=result.replace(/\\([a-zA-Z]+)\b/g,function(_,name){return symbols[name]||"";});return result.replace(/<[^>]+>/g,"").replace(/```[a-zA-Z0-9_-]*/g,"").replace(/`([^`]*)`/g,"$1").replace(/\*\*(.*?)\*\*/gs,"$1").replace(/__([\s\S]*?)__/g,"$1").replace(/^\s{0,3}#{1,6}\s+/gm,"").replace(/\$\$?/g,"").replace(/\\\[|\\\]|\\\(|\\\)/g,"").replace(/[{}]/g,"").replace(/\\left|\\right/g,"").replace(/\\,/g," ").replace(/\\%/g,"%").replace(/\\[a-zA-Z]+/g,"").replace(/[ \t]{2,}/g," ");}
function renderRichText(body,content){body.textContent=cleanLatex(content);}
function showCopyState(button,label){
  button.textContent=label;
  window.setTimeout(function(){button.textContent="Copy";},1200);
}
function fallbackCopy(text,button){
  let area=document.createElement("textarea");
  area.value=text;
  area.setAttribute("readonly","");
  area.style.position="fixed";
  area.style.opacity="0";
  document.body.appendChild(area);
  area.select();
  let copied=false;
  try{copied=document.execCommand("copy");}catch(error){copied=false;}
  area.remove();
  showCopyState(button,copied?"Copied":"Copy failed");
}
async function copyResponse(text,button){
  try{
    if(!navigator.clipboard||!window.isSecureContext){fallbackCopy(text,button);return;}
    await navigator.clipboard.writeText(text);
    showCopyState(button,"Copied");
  }catch(error){fallbackCopy(text,button);}
}
function addBubble(container,message,content,metaText){
  let article=document.createElement("article");
  article.className="bubble "+message.role;
  let meta=document.createElement("div");
  meta.className="meta";
  let metaLabel=document.createElement("span");
  metaLabel.textContent=metaText;
  meta.appendChild(metaLabel);
  article.appendChild(meta);
  let body=document.createElement("div");
  renderRichText(body,content);
  article.appendChild(body);
  if(message.role==="assistant"&&state.view==="response"){
    let copy=document.createElement("button");
    copy.type="button";
    copy.className="copy-button";
    copy.textContent="Copy";
    copy.title="Copy response";
    copy.onclick=function(){copyResponse(body.textContent,copy);};
    meta.appendChild(copy);
  }
  container.appendChild(article);
}


function renderChat(){renderRouter();let c=state.conversation;let container=byId("chat");if(!c||!c.messages.length){container.innerHTML="";let empty=document.createElement("div");empty.className="empty";empty.textContent="Start a planning session.";container.appendChild(empty);return;}container.innerHTML="";let foundThinking=false;if(state.view==="thinking"){c.messages.forEach(function(m){if(m.role!=="assistant")return;let parts=splitThinking(m.content,m.thinking,m.response);if(!parts.thinking)return;foundThinking=true;addBubble(container,m,parts.thinking,(m.provider||"assistant")+" | "+(m.paradigm||"")+" | thinking");});if(!foundThinking){let empty=document.createElement("div");empty.className="empty";empty.textContent="No thinking blocks were returned by the selected harness.";container.appendChild(empty);}}else{c.messages.forEach(function(m){if(m.role==="assistant"){let parts=splitThinking(m.content,m.thinking,m.response);addBubble(container,m,parts.response||"(thinking block only)",(m.provider||"assistant")+" | "+(m.paradigm||"")+" | response");}else{addBubble(container,m,m.content,"You | "+(m.paradigm||""));}});}container.scrollTop=container.scrollHeight;}
async function openConversation(id){state.conversation=await api("/api/conversations/"+encodeURIComponent(id));byId("provider").value=state.conversation.provider;byId("include-uploads").checked=false;renderModels();renderEfforts();renderPlanMode();renderUploads();renderSessions();renderChat();}
async function newConversation(){let title=window.prompt("Name this planning session","New planning session");if(title===null)return;title=title.trim()||"New planning session";let c=await api("/api/conversations",{method:"POST",body:JSON.stringify({title:title,provider:byId("provider").value,model:selectedModel(),effort:selectedEffort(),plan_mode:selectedPlanMode()})});state.boot.conversations.unshift({id:c.id,title:c.title,provider:c.provider,model:c.model,effort:c.effort,plan_mode:c.plan_mode,updated_at:c.updated_at,message_count:0,upload_count:0});state.conversation=c;byId("include-uploads").checked=false;renderModels();renderEfforts();renderPlanMode();renderUploads();renderSessions();renderChat();}
async function deleteConversation(){if(!state.conversation||state.sending)return;let currentId=state.conversation.id;if(window.confirm("Delete session: "+state.conversation.title+"?")===false)return;let button=byId("delete");button.disabled=true;try{await api("/api/conversations/"+encodeURIComponent(currentId),{method:"DELETE"});state.boot.conversations=state.boot.conversations.filter(function(c){return c.id!==currentId;});state.conversation=null;if(state.boot.conversations.length)await openConversation(state.boot.conversations[0].id);else{renderSessions();renderUploads();renderChat();}}catch(e){byId("warning").textContent=e.message;updateDeleteButton();}}
async function send(){let text=byId("message").value.trim();if(!text||!state.conversation||state.sending)return;let button=document.querySelector(".composer button");let conversationId=state.conversation.id;let runningMessage="Running the selected harness...";state.sending=true;button.disabled=true;button.textContent="Routing...";byId("warning").textContent=runningMessage;try{let result=await api("/api/conversations/"+encodeURIComponent(conversationId)+"/messages",{method:"POST",body:JSON.stringify({message:text,provider:byId("provider").value,model:selectedModel(),effort:selectedEffort(),plan_mode:selectedPlanMode(),paradigm:byId("paradigm").value,include_uploads:byId("include-uploads").checked})});if(state.conversation&&state.conversation.id===conversationId){state.conversation=result;byId("include-uploads").checked=false;}byId("message").value="";let c=state.boot.conversations.find(function(x){return x.id===conversationId;});if(c){c.message_count=result.messages.length;c.upload_count=(result.uploads||[]).length;c.updated_at=result.updated_at;c.provider=result.provider;c.model=result.model;c.effort=result.effort;c.plan_mode=result.plan_mode;}if(state.conversation&&state.conversation.id===conversationId){renderModels();renderEfforts();renderPlanMode();renderSessions();renderChat();}}catch(e){byId("warning").textContent=e.message;}finally{state.sending=false;button.disabled=false;button.textContent="Send";renderUploads();if(byId("warning").textContent===runningMessage)byId("warning").textContent="";}}
async function boot(){state.boot=await api("/api/bootstrap");renderProviders();renderSessions();if(state.boot.conversations.length)await openConversation(state.boot.conversations[0].id);else{renderUploads();renderChat();}}
byId("new").onclick=newConversation;byId("delete").onclick=deleteConversation;byId("upload-files").onchange=uploadFiles;byId("clear-uploads").onclick=clearUploads;byId("provider").onchange=function(){if(state.conversation){state.conversation.model="";}byId("custom-model").value="";renderModels();renderEfforts();renderPlanMode();};byId("model").onchange=function(){byId("custom-model").value="";renderEfforts();};byId("custom-model").oninput=function(){renderEfforts();};byId("composer").onsubmit=function(e){e.preventDefault();send();};document.querySelectorAll(".view-tab").forEach(function(tab){tab.onclick=function(){setView(tab.dataset.view);};});boot().catch(function(e){byId("warning").textContent=e.message;});
</script></body></html>''';


class RequestHandler(BaseHTTPRequestHandler):
    app: ChatApplication

    def _json(self, status: int, payload: Any) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 2_000_000:
            raise ValueError("request too large")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8") or "{}")


    def _multipart_files(self) -> list[tuple[str, bytes]]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid upload length") from exc
        if length <= 0:
            raise ValueError("empty upload request")
        if length > _limit("SOT_UPLOAD_MAX_REQUEST_BYTES", UPLOAD_MAX_REQUEST_BYTES):
            raise ValueError("upload request is too large")

        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            raise ValueError("uploads must use multipart/form-data")
        raw = self.rfile.read(length)
        newline = chr(13) + chr(10)
        envelope = (
            f"Content-Type: {content_type}{newline}"
            f"MIME-Version: 1.0{newline}{newline}"
        ).encode("utf-8") + raw
        message = BytesParser(policy=email_policy).parsebytes(envelope)
        if not message.is_multipart():
            raise ValueError("invalid multipart upload")

        files: list[tuple[str, bytes]] = []
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data" or part.get_filename() is None:
                continue
            files.append((str(part.get_filename()), part.get_payload(decode=True) or b""))
        if not files:
            raise ValueError("no files uploaded")
        return files

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/":
                encoded = INDEX_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            if path == "/api/bootstrap":
                self._json(HTTPStatus.OK, self.app.bootstrap())
                return
            prefix = "/api/conversations/"
            if path.startswith(prefix):
                conversation = self.app.store.load(path[len(prefix):])
                self._json(HTTPStatus.OK, conversation.as_dict())
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            prefix = "/api/conversations/"
            upload_suffix = "/uploads"
            if path.startswith(prefix) and path.endswith(upload_suffix):
                conversation_id = path[len(prefix):-len(upload_suffix)]
                files = self._multipart_files()
                self._json(HTTPStatus.OK, self.app.upload(conversation_id, files).as_dict())
                return

            payload = self._body()
            if path == "/api/conversations":
                self._json(HTTPStatus.CREATED, self.app.create(payload).as_dict())
                return
            suffix = "/messages"
            if path.startswith(prefix) and path.endswith(suffix):
                conversation_id = path[len(prefix):-len(suffix)]
                self._json(HTTPStatus.OK, self.app.send(conversation_id, payload).as_dict())
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except ProviderError as exc:
            self._json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        prefix = "/api/conversations/"
        try:
            upload_suffix = "/uploads"
            if path.startswith(prefix) and path.endswith(upload_suffix):
                conversation_id = path[len(prefix):-len(upload_suffix)]
                self._json(HTTPStatus.OK, self.app.clear_uploads(conversation_id).as_dict())
                return
            if path.startswith(prefix):
                self.app.delete(path[len(prefix):])
                self._json(HTTPStatus.OK, {"deleted": True})
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (OSError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        if os.environ.get("SOT_CHAT_QUIET") != "1":
            super().log_message(format, *args)


def serve(host: str = "127.0.0.1", port: int = 8787) -> None:
    app = ChatApplication()
    handler = type("SoTRequestHandler", (RequestHandler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"SoT chat listening at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the local Sketch-of-Thought planning chat")
    parser.add_argument("--host", default=os.environ.get("SOT_CHAT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SOT_CHAT_PORT", "8787")))
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()

