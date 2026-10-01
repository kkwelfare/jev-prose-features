"""Bounded, segment-bound advisory prose-feature classification for Jev."""
from __future__ import annotations
import json, math, re
from typing import Any, Mapping

TAXONOMY_VERSION = "jev-prose-features-1.5"
REVIEW_THRESHOLD = 0.5
REVIEW_FOCUS = {
 "japanese_naturalness": {"translationese": "直訳調に見える言い回しが、日本語として自然に読めるかを前後の流れと合わせて確認する。", "other_awkward": "言い回しや文のつながりが、前後の流れの中で自然に読めるかを確認する。"},
 "abstract_action_unclear": "行動の具体性が、目的や前後の説明を踏まえて読み手に伝わるかを確認する。",
 "redundant_paraphrase": "同じ内容の繰り返しが、前後の流れで理解を助けているか、読み進めにくさにつながっているかを確認する。",
 "unnecessary_contrast": "対比が、目的に照らして読み手の理解を助けているかを確認する。",
 "excessive_praise_empathy": "称賛や共感の表現が、読み手にとって自然で過不足ないかを確認する。",
 "empty_preamble_conclusion": {"preamble": "導入が本文への自然な入り口になっているかを確認する。", "conclusion": "結びが前後の流れを自然にまとめているかを確認する。"},
 "scope_expression": "範囲の説明が、読み手の理解に必要か、流れの中で自然かを確認する。",
}
BINARY_REVIEW_AXES = {"abstract_action_unclear", "redundant_paraphrase", "unnecessary_contrast", "excessive_praise_empathy"}
FEATURES = ("empty_preamble_conclusion", "abstract_action_unclear", "redundant_paraphrase", "unnecessary_contrast", "excessive_praise_empathy", "japanese_naturalness", "scope_expression", "author_style_impression")
# Keep choice text short; put definitions and exceptions in the axis instruction.
CATEGORICAL = {
 "empty_preamble_conclusion": ("preamble", "conclusion", "neither"),
 "japanese_naturalness": ("natural", "translationese", "other_awkward"),
 "scope_expression": ("none", "necessary_scope", "scope_only"),
 "author_style_impression": ("ai_like", "human_like"),
}
FEATURE_CRITERIA = {
 "empty_preamble_conclusion": {"preamble": "主に一般的な前置き", "conclusion": "主に一般的な結び", "neither": "どちらにも該当しない"},
 "abstract_action_unclear": {"present": "行動の具体性が不十分", "absent": "該当する特徴がない"},
 "redundant_paraphrase": {"present": "別の提示文と内容が重複", "absent": "該当する特徴がない"},
 "unnecessary_contrast": {"present": "不要な対比がある", "absent": "該当する特徴がない"},
 "excessive_praise_empathy": {"present": "称賛・共感が過剰", "absent": "該当する特徴がない"},
 "japanese_naturalness": {"natural": "日本語として自然に読める", "translationese": "直訳調の言い回しで不自然", "other_awkward": "直訳調以外の言い回しで不自然"},
 "scope_expression": {"none": "該当する範囲限定表現がない", "necessary_scope": "理解や判断に必要な対象・条件を限定", "scope_only": "既知の範囲を言い直し新情報なし"},
 "author_style_impression": {"ai_like": "AIが書いた文に見える。", "human_like": "人間が書いた文に見える。"},
}
FEATURE_INSTRUCTIONS = {
 "empty_preamble_conclusion": "Classify whether the target functions as generic preamble/conclusion, considering its purpose, neighboring context, and reader. Useful framing, introductions, recaps, and conclusions that support comprehension are not defects merely because they add no new facts; preserve reader orientation.",
 "abstract_action_unclear": "Assess whether the action is understandable to the intended reader in its purpose and surrounding flow. If prior or following context supplies an abstract action's referent, do not call it unclear. Preserve purposeful explanation and courtesy; flag only materially missing information that impedes comprehension.",
 "redundant_paraphrase": "Compare the target with supplied neighboring text and purpose. Repetition, recap, or restatement may support comprehension; do not treat lack of new information or a generic label alone as a defect or recommend removal. Flag only when the actual flow makes the repetition distracting or confusing.",
 "unnecessary_contrast": "Judge whether a contrast serves the reader's understanding in context and for the stated purpose. Preserve useful distinctions and purposeful emphasis; do not recommend removing a contrast merely because it adds no new fact.",
 "excessive_praise_empathy": "Assess tone in context of purpose and reader. Preserve ordinary courtesy and empathy that help the reader; flag only disproportionate or formulaic wording that gets in the way of comprehension.",
 "japanese_naturalness": "Judge whether wording reads naturally as Japanese beyond intelligibility, in the purpose, intended reader, and surrounding flow. Translationese is genuinely unnatural literal/calque-like wording, including awkward verb–object collocations, action/clause wording awkwardly used as a noun option label, or excessive nominalization, possessive/pronoun use, or literal word order when awkward for purpose and context. other_awkward means unnatural for another wording reason; do not merge ordinary action-clarity issues into either label. Do not infer translation origin, source language, truth, or authorship. Natural action proposals, menu/treatment choices, function names, and quoted technical labels are not translationese merely for using 「〜を選ぶ」. Select the best-supported available substantive option; express ambiguity through the returned confidence and probabilities.",
 "scope_expression": "Judge scope-limiting expressions only in context of purpose, surrounding text, and reader understanding. necessary_scope limits a subject or condition needed for reader understanding or judgment; scope_only restates an already-known scope without adding information. Do not mistake necessary qualifications, conditions, contrasts, or explanations for defects and do not recommend removing them merely because they add no new information. Select the best-supported available substantive option; express ambiguity through the returned confidence and probabilities.",
 "author_style_impression": "対象文と前後の文から、どちらの文体に見えるかを分類する。これは独立した印象軸であり、他の軸の評価や文章の良し悪しに結び付けない。",
}
MAX_TEXT_CHARS, MAX_SEGMENT_CHARS, MAX_SEGMENTS = 12000, 3000, 8
MAX_INPUT_BYTES, MAX_REQUEST_BYTES, MAX_TIMEOUT_SECONDS = 32000, 40000, 12.0
EVALUATION_OBJECTIVE = "読者が目的・読者像・前後の流れを踏まえ、日本語の文章として自然に読み進められるかを確認する。文章を短くしたり文を削ったりすること自体を目的にしない。内容・意図・語り口を保ち、読者が引っかかる言い回しやつながりを確認する。事実・内容の正しさ、英語由来か翻訳文かは推定・判定しない。8つの評価軸はそれぞれのcriteriaに従い、すべてを文法・自然さの問題にまとめない。"

class InputError(ValueError): pass

def _overlap(a,b): return a[0] < b[1] and b[0] < a[1]
def _protected(text):
 found=[]
 patterns=((r"(?m)^\s*(`{3,}|~{3,})[^\n]*(?:\n|$)[\s\S]*?^\s*\1[^\n]*(?:\n|$)","code"),(r"`[^`\n]*`","code"),(r"(?m)(?:^|[ \t])>[^\n]*","quote"),(r"“[^”\n]{1,2000}”|‘[^’\n]{1,2000}’|\"[^\"\n]{1,2000}\"|'[^'\n]{1,2000}'","quote"),(r"https?://[^\s<>]+","technical"))
 for pattern,kind in patterns:
  try: matches=re.finditer(pattern,text)
  except re.error: continue
  for m in matches:
   row={"start":m.start(),"end":m.end(),"kind":kind}
   if row["end"]>row["start"] and not any(_overlap((row["start"],row["end"]),(x["start"],x["end"])) for x in found): found.append(row)
 return sorted(found,key=lambda r:(r["start"],r["end"]))
def _prose(text,spans):
 parts=[]; pos=0
 for row in spans:
  if pos<row["start"]: parts.append(text[pos:row["start"]])
  pos=max(pos,row["end"])
 if pos<len(text) and text[pos:].strip(): parts.append(text[pos:])
 return re.sub(r"[「」『』]", "", "\n".join(p for p in parts if p.strip()))
def prepare_input(raw: Any)->dict[str,Any]:
 if not isinstance(raw,Mapping) or set(raw)-{"text","segments","purpose","context","reader"}: raise InputError("invalid_input")
 if ("text" in raw)==("segments" in raw): raise InputError("provide_text_or_segments")
 purpose,context,reader=raw.get("purpose",""),raw.get("context",""),raw.get("reader","")
 if not isinstance(purpose,str) or len(purpose)>240 or not isinstance(context,str) or len(context)>500 or not isinstance(reader,str) or len(reader)>240: raise InputError("invalid_context")
 rows=[]
 if "text" in raw:
  text=raw["text"]
  if not isinstance(text,str) or len(text)>MAX_TEXT_CHARS: raise InputError("input_too_large_or_invalid")
  for i,m in enumerate(re.finditer(r"\S(?:[\s\S]*?\S)?(?=\n\s*\n|$)",text),1):
   excerpt=m.group(); local=_protected(excerpt); shifted=[{**s,"start":s["start"]+m.start(),"end":s["end"]+m.start()} for s in local]
   rows.append({"id":f"p{i}","text":excerpt,"previous_sentence":"","next_sentence":"","start":m.start(),"end":m.end(),"offset_scope":"input_text","skipped":shifted,"classification_text":_prose(excerpt,local)})
  if not rows: rows=[{"id":"p1","text":"","previous_sentence":"","next_sentence":"","start":0,"end":0,"offset_scope":"input_text","skipped":[],"classification_text":""}]
 else:
  items=raw["segments"]
  if not isinstance(items,list) or not 1<=len(items)<=MAX_SEGMENTS: raise InputError("invalid_segments")
  ids=set()
  for item in items:
   if not isinstance(item,Mapping) or set(item)-{"id","text","previous_sentence","next_sentence"} or not {"id","text"}<=set(item): raise InputError("invalid_segments")
   sid,text=item["id"],item["text"]
   previous_sentence,next_sentence=item.get("previous_sentence",""),item.get("next_sentence","")
   if (not isinstance(sid,str) or not sid or len(sid)>128 or sid in ids or not isinstance(text,str) or len(text)>MAX_SEGMENT_CHARS
       or not isinstance(previous_sentence,str) or len(previous_sentence)>MAX_SEGMENT_CHARS
       or not isinstance(next_sentence,str) or len(next_sentence)>MAX_SEGMENT_CHARS): raise InputError("invalid_segments")
   ids.add(sid); spans=_protected(text); previous_spans=_protected(previous_sentence); next_spans=_protected(next_sentence)
   rows.append({"id":sid,"text":text,"previous_sentence":previous_sentence,"next_sentence":next_sentence,"previous_classification_text":_prose(previous_sentence,previous_spans),"next_classification_text":_prose(next_sentence,next_spans),"start":0,"end":len(text),"offset_scope":"segment_local","skipped":spans,"classification_text":_prose(text,spans)})
 if len(rows)>MAX_SEGMENTS or sum(sum(len(r[k].encode()) for k in ("text","previous_sentence","next_sentence")) for r in rows)>MAX_INPUT_BYTES: raise InputError("input_too_large")
 return {"purpose":purpose,"context":context,"reader":reader,"segments":rows,"input_kind":"text" if "text" in raw else "segments"}
def build_request(clean:Mapping[str,Any],*,model:str)->dict[str,Any]:
 rows=[{"id":r["id"],"text":r["classification_text"],"previous_sentence":r.get("previous_sentence",""),"next_sentence":r.get("next_sentence","")} for r in clean["segments"] if r["classification_text"].strip()]
 if not rows: raise InputError("no_unprotected_prose")
 state={"taxonomy_version":TAXONOMY_VERSION,"evaluation_objective":EVALUATION_OBJECTIVE,"purpose":clean["purpose"],"context":clean["context"],"reader":clean["reader"],"segments":rows}; questions={}; qmap={}
 for i,row in enumerate(rows):
  for feature in FEATURES:
   key=f"s{i}_{feature}"; labels=CATEGORICAL.get(feature,("present","absent"))
   criteria={label:FEATURE_CRITERIA[feature][label] for label in labels}
   context_rule=" Classify ONLY the target text for this segment; previous_sentence and next_sentence are interpretation context for references, omissions, and coherence, never additional text to classify or flag."
   if feature=="redundant_paraphrase": context_rule+=" You may compare the target proposition with supplied neighboring sentences for redundancy, but do not assess the neighbors as targets."
   questions[key]={"type":"choice","instructions":f"For segment id {row['id']!r}, classify only {feature}. {FEATURE_INSTRUCTIONS[feature]}{context_rule} This is advisory, not a quality verdict. Choose the best-supported available substantive option and express ambiguity using the returned confidence and probabilities. Do not propose rewrites or actions.","criteria":criteria}; qmap[key]=(row["id"],feature,labels)
 request={"model":model,"state":json.dumps(state,ensure_ascii=False,separators=(",",":")),"questions":questions}
 if len(request["state"].encode())>MAX_REQUEST_BYTES: raise InputError("request_too_large")
 request["_question_map"]=qmap
 return request
def _num(value):
 if isinstance(value,bool) or not isinstance(value,(int,float)): raise ValueError("invalid_confidence")
 n=float(value)
 if not math.isfinite(n) or not 0<=n<=1: raise ValueError("invalid_confidence")
 return n
def parse_response(value:Any,*,transport:str|None=None)->dict[str,Any]:
 if not isinstance(value,Mapping) or not isinstance(value.get("answers"),Mapping): raise ValueError("malformed_response")
 model=value.get("model","unknown")
 if not isinstance(model,str) or not model.strip(): raise ValueError("malformed_response")
 return {"model":model,"answers":value["answers"]}
def validate_response(clean,request,parsed):
 answers=parsed.get("answers"); qmap=request.get("_question_map")
 if not isinstance(answers,Mapping) or set(answers)!=set(qmap): raise ValueError("question_mismatch")
 output={r["id"]:{} for r in clean["segments"]}
 for key,(sid,feature,labels) in qmap.items():
  answer=answers[key]
  if not isinstance(answer,Mapping) or answer.get("choice") not in labels: raise ValueError("invalid_label")
  confidence=_num(answer.get("confidence")); probs=answer.get("probabilities")
  if not isinstance(probs,Mapping) or set(probs)!=set(labels): raise ValueError("invalid_probabilities")
  probs={k:_num(probs[k]) for k in labels}
  if abs(sum(probs.values())-1)>0.02: raise ValueError("invalid_probabilities")
  choice=answer["choice"]
  output[sid][feature]={"label":choice,"raw_label":choice,"confidence":confidence,"probabilities":probs}
 return {"model":parsed.get("model","unknown"),"features":output}
def _review_view(features):
 candidates=[]; assessed=False
 if not isinstance(features,Mapping):
  return {"review_threshold":REVIEW_THRESHOLD,"review_status":"unassessed","review_candidates":[],"review_confidence":None}
 for feature in FEATURES:
  value=features.get(feature)
  if not isinstance(value,Mapping): continue
  label=value.get("label",value.get("raw_label")); confidence=value.get("confidence")
  if label=="unassessed" or confidence is None: continue
  assessed=True
  try: confidence=_num(confidence)
  except (TypeError,ValueError): continue
  focus=None
  if feature=="japanese_naturalness" and label in {"translationese","other_awkward"} and confidence>=REVIEW_THRESHOLD:
   focus=REVIEW_FOCUS[feature][label]
  elif feature in BINARY_REVIEW_AXES and label=="present" and confidence>=REVIEW_THRESHOLD:
   focus=REVIEW_FOCUS[feature]
  elif feature=="empty_preamble_conclusion" and label in {"preamble","conclusion"} and confidence>=REVIEW_THRESHOLD:
   focus=REVIEW_FOCUS[feature][label]
  elif feature=="scope_expression" and label=="scope_only" and confidence>=REVIEW_THRESHOLD:
   focus=REVIEW_FOCUS[feature]
  if focus is not None:
   candidates.append({"feature":feature,"label":label,"confidence":confidence,"focus":focus})
 if candidates:
  candidates.sort(key=lambda item:(item["feature"]!="japanese_naturalness",FEATURES.index(item["feature"])))
 return {"review_threshold":REVIEW_THRESHOLD,"review_status":"assessed" if assessed else "unassessed","review_candidates":candidates,"review_confidence":max((item["confidence"] for item in candidates),default=None)}
def unknown_result(clean,reason):
 return {"status":"unknown","reason":reason,"taxonomy_version":TAXONOMY_VERSION,"advisory_only":True,"correctness_not_assessed":True,"segments":[{"id":r["id"],"start":r["start"],"end":r["end"],"offset_scope":r["offset_scope"],"excerpt":r["text"],"skipped":r["skipped"],"features":{f:{"label":"unassessed","confidence":None,"raw_label":None} for f in FEATURES},**_review_view({})} for r in clean["segments"]]}
def bind_response(clean,decision):
 features=decision.get("features",{}); segments=[]
 for r in clean["segments"]:
  found=features.get(r["id"],{}) if isinstance(features,Mapping) else {}
  if not isinstance(found,Mapping): found={}
  complete={feature:found.get(feature,{"label":"unassessed","confidence":None,"raw_label":None}) for feature in FEATURES}
  segments.append({"id":r["id"],"start":r["start"],"end":r["end"],"offset_scope":r["offset_scope"],"excerpt":r["text"],"skipped":r["skipped"],"features":complete,**_review_view(complete)})
 return {"status":"classified","taxonomy_version":TAXONOMY_VERSION,"advisory_only":True,"correctness_not_assessed":True,"confidence_is_not_correctness":True,"model":decision.get("model","unknown"),"segments":segments}
