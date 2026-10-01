"""Offline tests for source mapping, protection, response validation and route selection."""
from __future__ import annotations
import importlib.util, json, sys, unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("jev_prose_features_test_plugin",ROOT/"__init__.py",submodule_search_locations=[str(ROOT)])
plugin=importlib.util.module_from_spec(spec); sys.modules[spec.name]=plugin; spec.loader.exec_module(plugin)
core=sys.modules["_jev_prose_features_core"]
class Context:
 profile_name="ops"
 def __init__(self,config=None): self.tools={}; self.config=config or {}
 def get_config(self,k,d=None): return self.config.get(k,d)
 def register_tool(self,**kw): self.tools[kw["name"]]=kw
class CoreTests(unittest.TestCase):
 def test_sentence_context_validation_protection_and_source_binding(self):
  raw={"purpose":"記事の理解","context":"仕事の進め方を説明する","reader":"就職準備中の人","segments":[{"id":"target-1","text":"「確認する」と伝える。","previous_sentence":"前の文です。","next_sentence":"後の文です。"}]}
  clean=core.prepare_input(raw); row=clean["segments"][0]
  req=core.build_request(clean,model="fixture"); state=json.loads(req["state"])
  self.assertEqual(state["reader"],"就職準備中の人")
  self.assertEqual(state["segments"],[{"id":"target-1","text":"確認すると伝える。","previous_sentence":"前の文です。","next_sentence":"後の文です。"}])
  self.assertEqual((row["id"],row["text"],row["start"],row["end"]),("target-1",raw["segments"][0]["text"],0,len(raw["segments"][0]["text"])))
  self.assertEqual(row["skipped"],[])
  for q in req["questions"].values():
   self.assertIn("ONLY the target text",q["instructions"])
   self.assertIn("never additional text to classify or flag",q["instructions"])
  red=next(q for q in req["questions"].values() if "redundant_paraphrase" in q["instructions"])
  self.assertIn("compare the target proposition",red["instructions"])
  for invalid in (None,1,True,"x"*(core.MAX_SEGMENT_CHARS+1)):
   with self.assertRaises(core.InputError): core.prepare_input({"segments":[{"id":"x","text":"target","previous_sentence":invalid}]})
  with self.assertRaises(core.InputError): core.prepare_input({"segments":[{"id":"x","text":"target","unexpected":"x"}]})
  with self.assertRaises(core.InputError): core.prepare_input({"segments":[{"id":"x","text":"target","next_sentence":"n"*(core.MAX_SEGMENT_CHARS+1)}]})
  with self.assertRaises(core.InputError): core.prepare_input({"segments":[{"id":"x","text":"t","previous_sentence":"前"*17000,"next_sentence":"後"*17000}]})
  self.assertEqual(core.prepare_input({"segments":[raw["segments"][0]]})["segments"][0]["previous_sentence"],"前の文です。")
  for item in ({"id":"legacy","text":"本文"},{"id":"explicit","text":"本文","previous_sentence":"","next_sentence":""}):
   self.assertEqual(core.prepare_input({"segments":[item]})["segments"][0]["previous_sentence"],"")
  bracket=core.prepare_input({"segments":[{"id":"q","text":"対象「引用」末尾","previous_sentence":"前「引用」","next_sentence":"後『引用』"}]})
  br=bracket["segments"][0]
  self.assertEqual(br["text"],"対象「引用」末尾"); self.assertEqual((br["start"],br["end"]),(0,len(br["text"])))
  self.assertEqual(br["classification_text"],"対象引用末尾")
  contextual=core.prepare_input({"segments":[{"id":"q2","text":"target","previous_sentence":"前文 `code` と「引用」","next_sentence":"後文「括弧」『引用』"}]})
  self.assertEqual(contextual["segments"][0]["previous_classification_text"],"前文 \n と引用")
  self.assertEqual(contextual["segments"][0]["next_classification_text"],"後文括弧引用")
  contextual_state=json.loads(core.build_request(contextual,model="fixture")["state"])
  self.assertEqual(contextual_state["segments"][0]["text"],"target")
  self.assertEqual(contextual_state["segments"][0]["previous_sentence"],"前文 `code` と「引用」")
  self.assertEqual(contextual_state["segments"][0]["next_sentence"],"後文「括弧」『引用』")
 def test_schema_and_adapter_batch_wire_keep_context_and_only_target_questions(self):
  schema=plugin._schema()["parameters"]
  self.assertEqual(schema["properties"]["reader"]["maxLength"],240)
  segment_schema=schema["properties"]["segments"]["items"]
  self.assertEqual(segment_schema["properties"]["previous_sentence"]["maxLength"],core.MAX_SEGMENT_CHARS)
  self.assertEqual(segment_schema["required"],["id","text"])
  rows=[{"id":f"s{i}","text":f"target-{i}","previous_sentence":f"prev-{i}","next_sentence":f"next-{i}"} for i in range(8)]
  sent=[]
  def fake_send(ctx,bridge,body,timeout):
   sent.append(json.loads(body["state"]))
   self.assertEqual(len(body["questions"]),8*len(sent[-1]["segments"]))
   style_questions=[q for q in body["questions"].values() if q["criteria"]=={"ai_like":"AIが書いた文に見える。","human_like":"人間が書いた文に見える。"}]
   self.assertEqual(len(style_questions),len(sent[-1]["segments"]))
   self.assertTrue(all("対象文と前後の文から、どちらの文体に見えるかを分類する。" in q["instructions"] for q in style_questions))
   answers={}
   # Valid offline fixture response sized to each batch's question map.
   for key in body["questions"]:
    criteria=body["questions"][key]["criteria"]; choice=next(iter(criteria))
    answers[key]={"choice":choice,"confidence":0.9,"probabilities":{v:(1.0 if v==choice else 0.0) for v in criteria}}
   return {"model":"offline-fixture","answers":answers}
  with patch.object(plugin,"_send",side_effect=fake_send):
   out=json.loads(plugin.ProseFeatureTool(Context())({"segments":rows,"reader":"就職準備中の人"}))
  self.assertEqual(out["status"],"classified"); self.assertEqual(len(sent),2)
  self.assertEqual([r["id"] for batch in sent for r in batch["segments"]],[r["id"] for r in rows])
  for batch in sent:
   for row in batch["segments"]:
    original=rows[int(row["id"][1:])]
    self.assertEqual((row["text"],row["previous_sentence"],row["next_sentence"]),(original["text"],original["previous_sentence"],original["next_sentence"]))
  self.assertEqual([s["id"] for s in out["segments"]],[r["id"] for r in rows])
  self.assertEqual([s["excerpt"] for s in out["segments"]],[r["text"] for r in rows])
  self.assertEqual(out["segments"][0]["start"],0)
  self.assertEqual(out["segments"][0]["end"],len(rows[0]["text"]))
  sent.clear()
  nine=[{"id":f"s{i}","text":f"target-{i}"} for i in range(9)]
  result=json.loads(plugin.ProseFeatureTool(Context())({"segments":nine}))
  self.assertEqual(result["status"],"unknown"); self.assertEqual(result["reason"],"invalid_segments")
  self.assertEqual(sent,[])
 def test_text_mode_and_old_segment_mode_default_context_and_reader(self):
  text=core.prepare_input({"text":"第一文。\n\n第二文。"})
  self.assertEqual(text["reader"],"")
  self.assertTrue(all(r["previous_sentence"]==r["next_sentence"]=="" for r in text["segments"]))
  legacy=core.prepare_input({"segments":[{"id":"old","text":"そのまま。"}],"purpose":"目的","context":"背景"})
  self.assertEqual((legacy["purpose"],legacy["context"],legacy["reader"]),("目的","背景",""))
  self.assertEqual((legacy["segments"][0]["previous_sentence"],legacy["segments"][0]["next_sentence"]),("",""))
  with self.assertRaises(core.InputError): core.prepare_input({"segments":[{"id":"x","text":"文"}],"reader":"r"*241})
 def test_text_exact_unicode_offsets_repetition(self):

  text="同じ文。\n\n同じ文。"
  clean=core.prepare_input({"text":text})
  self.assertEqual([s["id"] for s in clean["segments"]],["p1","p2"])
  for s in clean["segments"]: self.assertEqual(text[s["start"]:s["end"]],s["text"])
  self.assertEqual(clean["segments"][1]["start"],6)
 def test_protected_spans_excluded_and_recorded(self):
  clean=core.prepare_input({"segments":[{"id":"s","text":"通常の説明 `secret_code` と引用“quoted words”末尾"}]})
  request=core.build_request(clean,model="jev-latest")
  wire=json.dumps(json.loads(request["state"]),ensure_ascii=False)
  self.assertNotIn("secret_code",wire)
  self.assertNotIn("quoted words",wire)
  self.assertEqual([s["kind"] for s in clean["segments"][0]["skipped"]], ["code","quote"])
 def test_inline_protection_does_not_drop_following_prose(self):
  text="前置き > quote\n後続です `code` 続き"
  row=core.prepare_input({"text":text})["segments"][0]
  self.assertEqual(row["text"],text)
  self.assertIn("後続です",row["classification_text"])
  self.assertEqual([x["kind"] for x in row["skipped"]],["quote","code"])
 def test_japanese_brackets_keep_prose_but_preserve_protection_and_offsets(self):
  article="そこで、「今日は応募まで進めよう」と一度に決める代わりに、「求人を一件だけ開いて、気になったことを一つメモするならどうだろう」と考えてみます"
  raw="前置き\r\n\r\n"+article+"\r\n> 引用“protected block”\r\n`code「対象外」`\r\n『完全に括弧内』"
  clean=core.prepare_input({"segments":[{"id":"s","text":raw}]})
  row=clean["segments"][0]
  self.assertEqual(row["text"],raw)
  self.assertEqual((row["start"],row["end"],row["offset_scope"]),(0,len(raw),"segment_local"))
  self.assertEqual(raw[row["start"]:row["end"]],row["text"])
  self.assertIn("今日は応募まで進めよう",row["classification_text"])
  self.assertIn("求人を一件だけ開いて、気になったことを一つメモするならどうだろう",row["classification_text"])
  self.assertNotRegex(row["classification_text"],r"[「」『』]")
  self.assertNotIn("引用",row["classification_text"])
  self.assertNotIn("code",row["classification_text"])
  self.assertEqual([s["kind"] for s in row["skipped"]],["quote","code"])
  paragraph_text="前置き\n\n"+article+"\n\n後続"
  paragraph_rows=core.prepare_input({"text":paragraph_text})["segments"]
  self.assertEqual(len(paragraph_rows),3)
  self.assertEqual(paragraph_rows[1]["start"],paragraph_text.index(article))
  self.assertEqual(paragraph_text[paragraph_rows[1]["start"]:paragraph_rows[1]["end"]],article)
  self.assertEqual(paragraph_rows[1]["classification_text"],article.translate(str.maketrans("","","「」『』")))
  nested=core.prepare_input({"segments":[{"id":"nested","text":"『外「内」外』、「二つ」「続けて」"},{"id":"only","text":"「完全に括弧内」"}]})
  self.assertEqual(nested["segments"][0]["classification_text"],"外内外、二つ続けて")
  self.assertEqual(nested["segments"][1]["classification_text"],"完全に括弧内")
  self.assertEqual(json.loads(core.build_request(nested,model="fixture")["state"])["segments"],[{"id":"nested","text":"外内外、二つ続けて","previous_sentence":"","next_sentence":""},{"id":"only","text":"完全に括弧内","previous_sentence":"","next_sentence":""}])
  self.assertTrue(all(seg["end"]==len(seg["text"]) for seg in nested["segments"]))
 def test_explicit_stable_ids_and_local_offsets(self):

  with self.assertRaises(core.InputError): core.prepare_input({"segments":[{"id":"x","text":"a"},{"id":"x","text":"b"}]})
  row=core.prepare_input({"segments":[{"id":"s-1","text":"文字列"}]})["segments"][0]
  self.assertEqual((row["id"],row["start"],row["end"],row["offset_scope"]),("s-1",0,3,"segment_local"))
 def test_taxonomy_1_5_has_abstention_free_eight_axis_criteria_and_source_bound_choices(self):
  awkward="「何が気になっているかを、相談できる相手に話す」を選ぶ"
  natural="不安なことを相談できる相手に話してみる方法もあります"
  menu="治療法の一覧から、自分に合うものを選ぶ"
  quoted='関数名「対象を選ぶ」を呼び出します'
  raw=awkward+"。"+natural+"。"+menu+"。"+quoted
  clean=core.prepare_input({"segments":[{"id":"sentence-ja-17","text":raw}],"purpose":"相談先を案内する文","context":"選択肢の説明に続く文章"})
  row=clean["segments"][0]
  req=core.build_request(clean,model="fixture")
  state=json.loads(req["state"])
  feature="japanese_naturalness"
  key=next(k for k,(_,f,_) in req["_question_map"].items() if f==feature)
  criteria=req["questions"][key]["criteria"]
  self.assertEqual(core.TAXONOMY_VERSION,"jev-prose-features-1.5")
  self.assertEqual(len(core.FEATURES),8)
  self.assertNotIn("formulaic_enumeration",core.FEATURES)
  self.assertNotIn("japanese_translationese",core.FEATURES)
  self.assertEqual(set(core.FEATURES),{"empty_preamble_conclusion","abstract_action_unclear","redundant_paraphrase","unnecessary_contrast","excessive_praise_empathy","japanese_naturalness","scope_expression","author_style_impression"})
  self.assertEqual(state["segments"],[{"id":"sentence-ja-17","text":row["classification_text"],"previous_sentence":"","next_sentence":""}])
  self.assertIn("何が気になっているかを、相談できる相手に話す",state["segments"][0]["text"])
  self.assertIn(natural,state["segments"][0]["text"])
  self.assertIn(menu,state["segments"][0]["text"])
  self.assertIn("関数名対象を選ぶを呼び出します",state["segments"][0]["text"])
  self.assertEqual(criteria,{"natural":"日本語として自然に読める","translationese":"直訳調の言い回しで不自然","other_awkward":"直訳調以外の言い回しで不自然"})
  self.assertNotIn("uncertain",criteria)
  self.assertTrue(all("uncertain" not in q["instructions"] and "abstain" not in q["instructions"].lower() for q in req["questions"].values()))
  self.assertTrue(all(len(c)<=80 for q in req["questions"].values() for c in q["criteria"].values()))
  self.assertIn("noun option label",req["questions"][key]["instructions"])
  self.assertIn("menu/treatment choices",req["questions"][key]["instructions"])
  self.assertIn("source language",req["questions"][key]["instructions"])

  self.assertIn("confidence and probabilities",req["questions"][key]["instructions"])
  self.assertEqual(row["text"],raw)
  self.assertEqual((row["start"],row["end"]),(0,len(raw)))
  self.assertEqual(raw[row["start"]:row["end"]],row["text"])
  answers={}
  for question,(_,f,labels) in req["_question_map"].items():
   selected="natural" if f==feature else (labels[0] if f in {"empty_preamble_conclusion","author_style_impression"} else ("none" if f=="scope_expression" else "absent"))
   answers[question]={"choice":selected,"confidence":0.4 if f==feature else 0.9,"probabilities":{label:(1.0 if label==selected else 0.0) for label in labels}}
  validated=core.validate_response(clean,req,core.parse_response({"model":"fixture","answers":answers}))
  bound=core.bind_response(clean,validated)
  out=bound["segments"][0]
  self.assertEqual(out["id"],"sentence-ja-17")
  self.assertEqual(out["excerpt"],raw)
  self.assertEqual(out["features"][feature]["label"],"natural")
  self.assertEqual(out["features"][feature]["raw_label"],"natural")
  self.assertEqual(out["features"][feature]["confidence"],0.4)
  self.assertEqual(set(out["features"][feature]["probabilities"]),{"natural","translationese","other_awkward"})
  self.assertEqual(out["features"][feature]["raw_label"],"natural")
  unknown=core.unknown_result(clean,"provider_unavailable")
  self.assertEqual(unknown["segments"][0]["features"][feature]["label"],"unassessed")
  self.assertEqual(set(unknown["segments"][0]["features"]),set(core.FEATURES))
  self.assertTrue(all(v["label"]=="unassessed" and v["raw_label"] is None and v["confidence"] is None for v in unknown["segments"][0]["features"].values()))
  scope_key=next(k for k,(_,f,_) in req["_question_map"].items() if f=="scope_expression")
  scope_q=req["questions"][scope_key]
  self.assertEqual(scope_q["criteria"],{"none":"該当する範囲限定表現がない","necessary_scope":"理解や判断に必要な対象・条件を限定","scope_only":"既知の範囲を言い直し新情報なし"})
  self.assertIn("already-known scope",scope_q["instructions"])
  self.assertNotIn("preamble",scope_q["instructions"])
 def test_naturalness_and_scope_choices_are_source_bound_and_offline(self):
  cases=[("自然な文です。","japanese_naturalness","natural"),("直近の申請に限って確認します。","scope_expression","necessary_scope")]
  for text,feature,choice in cases:
   clean=core.prepare_input({"segments":[{"id":"bound-ja","text":text}],"purpose":"手順の案内","context":"直近の申請を対象にする"})
   req=core.build_request(clean,model="fixture")
   key=next(k for k,(_,f,_) in req["_question_map"].items() if f==feature)
   labels=req["_question_map"][key][2]
   self.assertIn(choice,labels)
   answers={}
   for q,(_,f,ls) in req["_question_map"].items():
    selected=choice if f==feature else ("preamble" if f=="empty_preamble_conclusion" else ("none" if "none" in ls else ("absent" if "absent" in ls else ls[0])))
    answers[q]={"choice":selected,"confidence":0.94,"probabilities":{label:(1.0 if label==selected else 0.0) for label in ls}}
   parsed=core.parse_response({"model":"fixture","answers":answers})
   result=core.bind_response(clean,core.validate_response(clean,req,parsed))
   segment=result["segments"][0]
   self.assertEqual(segment["excerpt"],text)
   self.assertEqual(segment["id"],"bound-ja")
   self.assertEqual(segment["features"][feature]["label"],choice)
 def test_author_style_impression_exact_adapter_contract_and_low_confidence(self):
  clean=core.prepare_input({"segments":[{"id":"style-ja","text":"確認して対応します。","previous_sentence":"必要な点を確認してください。","next_sentence":"結果を共有します。"}]})
  req=core.build_request(clean,model="fixture")
  key=next(k for k,(_,feature,_) in req["_question_map"].items() if feature=="author_style_impression")
  question=req["questions"][key]
  self.assertEqual(question["criteria"],{"ai_like":"AIが書いた文に見える。","human_like":"人間が書いた文に見える。"})
  self.assertTrue(question["instructions"].split(" Classify ONLY the target text",1)[0].split(". ",2)[-1].startswith("対象文と前後の文から、どちらの文体に見えるかを分類する。"))
  self.assertIn("ONLY the target text",question["instructions"])
  self.assertIn("previous_sentence and next_sentence are interpretation context",question["instructions"])
  self.assertNotIn("実際の作者",question["instructions"])
  self.assertNotIn("作者を特定",question["instructions"])
  state=json.loads(req["state"])
  self.assertNotIn("AI著者性は推定・判定しない",state["evaluation_objective"])
  self.assertIn("事実・内容の正しさ",state["evaluation_objective"])
  self.assertIn("英語由来か翻訳文か",state["evaluation_objective"])
  labels=("ai_like","human_like")
  for choice in labels:
   answers={}
   for q,(_,feature,available) in req["_question_map"].items():
    selected=choice if feature=="author_style_impression" else available[0]
    confidence=0.18 if feature=="author_style_impression" else 0.91
    answers[q]={"choice":selected,"confidence":confidence,"probabilities":{label:(1.0 if label==selected else 0.0) for label in available}}
   parsed=core.parse_response({"model":"fixture","answers":answers})
   validated=core.validate_response(clean,req,parsed)
   bound=core.bind_response(clean,validated)
   observed=bound["segments"][0]["features"]["author_style_impression"]
   self.assertEqual(observed["label"],choice)
   self.assertEqual(observed["raw_label"],choice)
   self.assertEqual(observed["confidence"],0.18)
   self.assertEqual(set(observed["probabilities"]),set(labels))
  unknown=core.unknown_result(clean,"provider_unavailable")
  unassessed=unknown["segments"][0]["features"]["author_style_impression"]
  self.assertEqual(unassessed,{"label":"unassessed","confidence":None,"raw_label":None})
  self.assertEqual(core.CATEGORICAL["author_style_impression"],labels)
 def test_fixture_contract_and_invalid_unknown(self):
  clean=core.prepare_input({"segments":[{"id":"s","text":"文"}]}); req=core.build_request(clean,model="jev-latest"); qmap=req.pop("_question_map")
  answers={}
  for key,(_,_,labels) in qmap.items():
   choice=labels[0]; answers[key]={"choice":choice,"confidence":0.91,"probabilities":{x:(1.0 if x==choice else 0.0) for x in labels}}
  parsed=core.parse_response({"model":"fixture","answers":answers}); val=core.validate_response(clean,{**req,"_question_map":qmap},parsed)
  self.assertEqual(core.bind_response(clean,val)["segments"][0]["id"],"s")
  answers[next(iter(answers))]["confidence"]=2
  with self.assertRaises(ValueError): core.validate_response(clean,{**req,"_question_map":qmap},core.parse_response({"model":"fixture","answers":answers}))
  unknown=core.unknown_result(clean,"provider_unavailable")
  self.assertEqual(unknown["status"],"unknown"); self.assertTrue(all(x["label"]=="unassessed" and x["raw_label"] is None and x["confidence"] is None for x in unknown["segments"][0]["features"].values()))
 def test_low_confidence_retains_provider_choice_and_categorical_rationale(self):
  clean=core.prepare_input({"segments":[{"id":"s","text":"Useful framing explains why this matters."}]})
  req=core.build_request(clean,model="jev-latest")
  categorical=next(k for k,(_,feature,_) in req["_question_map"].items() if feature=="empty_preamble_conclusion")
  criteria=req["questions"][categorical]["criteria"]
  self.assertTrue(all("一般的" in value or "該当" in value or "情報不足" in value for value in criteria.values()))
  qmap=req["_question_map"]; answers={}
  for key,(_,feature,labels) in qmap.items():
   choice=labels[0] if feature in {"empty_preamble_conclusion","author_style_impression"} else ("none" if feature=="scope_expression" else ("natural" if feature=="japanese_naturalness" else "absent"))
   answers[key]={"choice":choice,"confidence":0.42,"probabilities":{label:(1.0 if label==choice else 0.0) for label in labels}}
  parsed=core.parse_response({"model":"fixture","answers":answers})
  val=core.validate_response(clean,req,parsed)
  feature=val["features"]["s"]["japanese_naturalness"]
  self.assertEqual(feature["label"],"natural")
  self.assertEqual(feature["raw_label"],"natural")
  self.assertEqual(feature["confidence"],0.42)
 def test_low_confidences_and_zero_confidence_preserve_labels_and_probabilities(self):
  clean=core.prepare_input({"segments":[{"id":"s","text":"Prose."}]}); req=core.build_request(clean,model="fixture")
  for confidence in (0.0,0.18,0.54):
   answers={}
   for key,(_,feature,labels) in req["_question_map"].items():
    choice=labels[0]
    answers[key]={"choice":choice,"confidence":confidence,"probabilities":{label:(1.0 if label==choice else 0.0) for label in labels}}
   result=core.validate_response(clean,req,core.parse_response({"model":"fixture","answers":answers}))
   for feature,fields in result["features"]["s"].items():
    self.assertEqual(fields["label"],fields["raw_label"])
    self.assertEqual(fields["confidence"],confidence)
    self.assertEqual(sum(fields["probabilities"].values()),1.0)
 def test_old_uncertain_choices_and_probability_maps_are_rejected(self):
  clean=core.prepare_input({"segments":[{"id":"s","text":"Prose."}]}); req=core.build_request(clean,model="fixture")
  key=next(iter(req["_question_map"]))
  answers={k:{"choice":ls[0],"confidence":0.8,"probabilities":{label:(1.0 if label==ls[0] else 0.0) for label in ls}} for k,(_,_,ls) in req["_question_map"].items()}
  answers[key]["choice"]="uncertain"
  with self.assertRaisesRegex(ValueError,"invalid_label"):
   core.validate_response(clean,req,core.parse_response({"model":"fixture","answers":answers}))
  answers[key]={"choice":req["_question_map"][key][2][0],"confidence":0.8,"probabilities":{"uncertain":1.0}}
  with self.assertRaisesRegex(ValueError,"invalid_probabilities"):
   core.validate_response(clean,req,core.parse_response({"model":"fixture","answers":answers}))
 def test_missing_answers_fail_and_protected_only_rows_are_unassessed(self):
  clean=core.prepare_input({"segments":[{"id":"prose","text":"Readable prose."},{"id":"protected","text":"`code only`"}]})
  req=core.build_request(clean,model="fixture")
  answers={}
  for key,(_,_,labels) in req["_question_map"].items():
   choice=labels[0]; answers[key]={"choice":choice,"confidence":0.18,"probabilities":{label:(1.0 if label==choice else 0.0) for label in labels}}
  with self.assertRaisesRegex(ValueError,"question_mismatch"):
   core.validate_response(clean,req,core.parse_response({"model":"fixture","answers":dict(list(answers.items())[:-1])}))
  decision=core.validate_response(clean,req,core.parse_response({"model":"fixture","answers":answers}))
  bound=core.bind_response(clean,decision)
  self.assertEqual(bound["segments"][0]["features"]["japanese_naturalness"]["label"],"natural")
  protected=bound["segments"][1]
  self.assertEqual(protected["skipped"][0]["kind"],"code")
  self.assertTrue(all(f["label"]=="unassessed" and f["confidence"] is None and f["raw_label"] is None for f in protected["features"].values()))
  only=core.prepare_input({"segments":[{"id":"only","text":"`code only`"}]})
  with self.assertRaisesRegex(core.InputError,"no_unprotected_prose"):
   core.build_request(only,model="fixture")
  unknown=core.unknown_result(only,"provider_unavailable")
  self.assertTrue(all(f["label"]=="unassessed" and f["confidence"] is None and f["raw_label"] is None for f in unknown["segments"][0]["features"].values()))
 def test_caps_and_registration(self):
  with self.assertRaises(core.InputError): core.prepare_input({"text":"x"*(core.MAX_TEXT_CHARS+1)})
  with self.assertRaises(core.InputError): core.prepare_input({"segments":[{"id":"x","text":"x"*(core.MAX_SEGMENT_CHARS+1)}]})
  too_many="\n\n".join("part" for _ in range(core.MAX_SEGMENTS+1))
  with self.assertRaises(core.InputError): core.prepare_input({"text":too_many})
  c=Context(); plugin.register(c); self.assertIn("jev_classify_prose_features",c.tools)
 def test_openrouter_primary_is_supported_config(self):
  cfg={"primary_provider":"openrouter"}
  p,timeout=plugin._provider_config(Context(cfg))
  self.assertEqual((p["name"],p["endpoint"],p["model"],p["env"]),("openrouter","https://openrouter.ai/api/alpha/decisions","typesafe/jev-1.13","OPENROUTER_API_KEY"))
 def test_available_openrouter_key_selected_without_typesafe_key(self):
  original=plugin._runtime_key
  for values,expected in [({"TYPESAFE_API_KEY":"","OPENROUTER_API_KEY":"synthetic-test-only"},"openrouter"),
                          ({"TYPESAFE_API_KEY":"typesafe-test","OPENROUTER_API_KEY":"openrouter-test"},"typesafe"),
                          ({"TYPESAFE_API_KEY":"","OPENROUTER_API_KEY":""},"typesafe")]:
   def fake_key(name): return values[name]
   plugin._runtime_key=fake_key
   try: p,_=plugin._provider_config(Context())
   finally: plugin._runtime_key=original
   self.assertEqual(p["name"],expected)
 def test_dotenv_loader_reads_only_requested_credential(self):
  import os, tempfile
  original_env=os.environ.get("HERMES_HOME")
  original_key=os.environ.pop("JEV_TEST_KEY_7F31",None)
  with tempfile.TemporaryDirectory() as tmp:
   from pathlib import Path
   Path(tmp,".env").write_text("JEV_TEST_KEY_7F31='synthetic-secret'\nOTHER_KEY=ignored\n")
   os.environ["HERMES_HOME"]=tmp
   try: self.assertEqual(plugin._runtime_key("JEV_TEST_KEY_7F31"),"synthetic-secret")
   finally:
    os.environ.pop("JEV_TEST_KEY_7F31",None)
    if original_key is not None: os.environ["JEV_TEST_KEY_7F31"]=original_key
    if original_env is None: os.environ.pop("HERMES_HOME",None)
    else: os.environ["HERMES_HOME"]=original_env
 def test_unavailable_route_returns_unknown(self):
  with patch.object(plugin,"_send",side_effect=RuntimeError("credential_unavailable")):
   out=json.loads(plugin.ProseFeatureTool(Context())({"segments":[{"id":"s","text":"hello"}]}))
  self.assertEqual(out["status"],"unknown"); self.assertNotEqual(out["segments"][0]["features"]["abstract_action_unclear"]["label"],"absent")
 def test_request_objective_present_with_and_without_context_and_caller_values_unchanged(self):
  for raw in (
   {"segments":[{"id":"empty-context","text":"確認して対応します。"}],"purpose":"依頼文"},
   {"segments":[{"id":"supplied-context","text":"確認して対応します。"}],"purpose":"依頼文","context":"相手に次の手順を案内する"},
  ):
   clean=core.prepare_input(raw)
   before={key:clean[key] for key in ("purpose","context")}
   state=json.loads(core.build_request(clean,model="fixture")["state"])
   self.assertEqual(state["evaluation_objective"],core.EVALUATION_OBJECTIVE)
   self.assertEqual({key:state[key] for key in before},before)
   self.assertEqual(state["segments"],[{"id":raw["segments"][0]["id"],"text":clean["segments"][0]["classification_text"],"previous_sentence":"","next_sentence":""}])
 def test_request_uses_decisions_contract_and_provider_failure_unknown(self):
  clean=core.prepare_input({"segments":[{"id":"ja-1","text":"確認して対応します。"}],"purpose":"業務メモ"})
  req=core.build_request(clean,model="typesafe/jev-1.13")
  self.assertEqual(set(req),{"model","state","questions","_question_map"})
  state=json.loads(req["state"])
  self.assertEqual(state["evaluation_objective"],core.EVALUATION_OBJECTIVE)
  self.assertIn("日本語の文章として自然に読み進められるか",state["evaluation_objective"])
  self.assertEqual((state["purpose"],state["context"]),("業務メモ",""))
  self.assertEqual(set(state),{"taxonomy_version","evaluation_objective","purpose","context","reader","segments"})
  self.assertEqual(set(req["questions"]),set(req["_question_map"]))
  self.assertTrue(all(q["type"]=="choice" for q in req["questions"].values()))
  class BrokenContext(Context):
   def get_config(self,k,d=None):
    return {"primary_provider":"openrouter"}.get(k,d)
  with patch.object(plugin,"_send",side_effect=RuntimeError("network_error")):
   out=json.loads(plugin.ProseFeatureTool(BrokenContext())({"segments":[{"id":"x","text":"synthetic"}]}))
  self.assertEqual(out["status"],"unknown")
  self.assertEqual(out["status"],"unknown"); self.assertTrue(all(f["label"]=="unassessed" and f["confidence"] is None and f["raw_label"] is None for f in out["segments"][0]["features"].values()))
 def test_tool_classified_response_is_fixture_only(self):
  clean=core.prepare_input({"segments":[{"id":"fixture-ja","text":"具体的な次の行動を確認する。"}]})
  req=core.build_request(clean,model="typesafe/jev-1.13")
  fixture={}
  for key,(_,_,labels) in req["_question_map"].items():
   choice="absent" if "absent" in labels else ("neither" if "neither" in labels else labels[0])
   fixture[key]={"choice":choice,"confidence":0.88,"probabilities":{label:(1.0 if label==choice else 0.0) for label in labels}}
  with patch.object(plugin,"_send",return_value={"model":"jev-fixture","answers":fixture}):
   result=json.loads(plugin.ProseFeatureTool(Context())({"segments":[{"id":"fixture-ja","text":"具体的な次の行動を確認する。"}]}))
  self.assertEqual(result["status"],"classified")
  self.assertEqual(result["segments"][0]["id"],"fixture-ja")
  self.assertTrue(result["advisory_only"])
  self.assertEqual(result["segments"][0]["features"]["abstract_action_unclear"]["label"],"absent")

 def test_review_candidates_are_contextual_and_never_replace_raw_axis_results(self):
  clean=core.prepare_input({"segments":[{"id":"review-ja","text":"対象文です。","previous_sentence":"前の流れ。","next_sentence":"次の流れ。"}],"purpose":"説明の自然さ","reader":"初めて読む人"})
  baseline={feature:{"label":("absent" if feature in core.BINARY_REVIEW_AXES else "natural" if feature=="japanese_naturalness" else "neither" if feature=="empty_preamble_conclusion" else "none" if feature=="scope_expression" else "human_like"),"raw_label":("absent" if feature in core.BINARY_REVIEW_AXES else "natural" if feature=="japanese_naturalness" else "neither" if feature=="empty_preamble_conclusion" else "none" if feature=="scope_expression" else "human_like"),"confidence":0.8,"probabilities":{"chosen":0.7,"other":0.3}} for feature in core.FEATURES}
  def bind(changes):
   features={key:{**baseline[key],**value} for key,value in changes.items()}
   return core.bind_response(clean,{"model":"offline-fixture","features":{"review-ja":features}})["segments"][0]
  natural=bind({"japanese_naturalness":{"label":"natural","raw_label":"natural","confidence":0.0},"author_style_impression":{"label":"ai_like","raw_label":"ai_like","confidence":1.0}})
  self.assertEqual(natural["review_candidates"],[])
  self.assertEqual(natural["review_status"],"assessed")
  self.assertIsNone(natural["review_confidence"])
  self.assertEqual(natural["features"]["japanese_naturalness"]["confidence"],0.0)
  self.assertEqual(natural["features"]["author_style_impression"]["confidence"],1.0)
  low=bind({"japanese_naturalness":{"label":"other_awkward","raw_label":"other_awkward","confidence":0.49}})
  self.assertEqual(low["review_candidates"],[])
  boundary=bind({"japanese_naturalness":{"label":"other_awkward","raw_label":"other_awkward","confidence":0.5},"author_style_impression":{"label":"ai_like","raw_label":"ai_like","confidence":1.0}})
  self.assertEqual(boundary["review_threshold"],0.5)
  self.assertEqual([(c["feature"],c["label"],c["confidence"]) for c in boundary["review_candidates"]],[("japanese_naturalness","other_awkward",0.5)])
  candidate=boundary["review_candidates"][0]
  self.assertIn("自然に読めるか",candidate["focus"])
  self.assertNotRegex(candidate["focus"],r"削除|短く|書き換え")
  self.assertEqual(boundary["features"]["japanese_naturalness"]["raw_label"],"other_awkward")
  self.assertEqual(boundary["features"]["japanese_naturalness"]["probabilities"],{"chosen":0.7,"other":0.3})
  self.assertEqual(boundary["review_confidence"],0.5)
  necessary=bind({"scope_expression":{"label":"necessary_scope","raw_label":"necessary_scope","confidence":1.0},"empty_preamble_conclusion":{"label":"neither","raw_label":"neither","confidence":1.0},"abstract_action_unclear":{"label":"absent","raw_label":"absent","confidence":1.0}})
  self.assertEqual(necessary["review_candidates"],[])
  present=bind({"redundant_paraphrase":{"label":"present","raw_label":"present","confidence":0.5}})
  self.assertEqual([c["feature"] for c in present["review_candidates"]],["redundant_paraphrase"])
  self.assertIn("理解を助けているか",present["review_candidates"][0]["focus"])
  for feature,label in (("japanese_naturalness","translationese"),("abstract_action_unclear","present"),("unnecessary_contrast","present"),("excessive_praise_empathy","present"),("empty_preamble_conclusion","preamble"),("empty_preamble_conclusion","conclusion"),("scope_expression","scope_only")):
   selected=bind({feature:{"label":label,"raw_label":label,"confidence":0.5}})
   self.assertEqual([(item["feature"],item["label"]) for item in selected["review_candidates"]],[(feature,label)])
  self.assertEqual(core.TAXONOMY_VERSION,"jev-prose-features-1.5")
 def test_review_candidates_unknown_and_protected_only_are_unassessed(self):
  for clean in (core.prepare_input({"segments":[{"id":"x","text":"本文"}]}),core.prepare_input({"segments":[{"id":"protected","text":"`code only`"}]})):
   result=core.unknown_result(clean,"provider_unavailable")
   segment=result["segments"][0]
   self.assertEqual(segment["review_threshold"],0.5)
   self.assertEqual(segment["review_status"],"unassessed")
   self.assertEqual(segment["review_candidates"],[])
   self.assertIsNone(segment["review_confidence"])
  protected=core.prepare_input({"segments":[{"id":"protected","text":"`code only`"},{"id":"prose","text":"本文"}]})
  req=core.build_request(protected,model="fixture")
  features={}
  for key,(sid,feature,labels) in req["_question_map"].items():
   choice=labels[0] if feature=="author_style_impression" else ("natural" if feature=="japanese_naturalness" else "neither" if feature=="empty_preamble_conclusion" else "none" if feature=="scope_expression" else "absent")
   features.setdefault(sid,{})[feature]={"label":choice,"raw_label":choice,"confidence":0.9,"probabilities":{label:(1.0 if label==choice else 0.0) for label in labels}}
  bound=core.bind_response(protected,{"model":"fixture","features":features})
  self.assertEqual(bound["segments"][0]["review_status"],"unassessed")
  self.assertEqual(bound["segments"][0]["review_candidates"],[])
  self.assertIsNone(bound["segments"][0]["review_confidence"])
  self.assertEqual(bound["segments"][1]["review_status"],"assessed")
  self.assertEqual(bound["segments"][1]["review_candidates"],[])
  self.assertIsNone(bound["segments"][1]["review_confidence"])
