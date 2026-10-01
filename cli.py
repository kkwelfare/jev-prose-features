"""Small JSON-in/JSON-out CLI for the on-demand Jev tool.

Input is one JSON object matching README. Use --validate-only for offline checks.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("jev_prose_features_cli_plugin",ROOT/"__init__.py",submodule_search_locations=[str(ROOT)])
if spec is None or spec.loader is None: raise SystemExit("plugin import failed")
plugin=importlib.util.module_from_spec(spec); sys.modules[spec.name]=plugin; spec.loader.exec_module(plugin)
core=sys.modules["_jev_prose_features_core"]
class Context:
 def get_config(self,key,default=None): return default

def main():
 parser=argparse.ArgumentParser(description="Advisory Jev prose feature classification")
 parser.add_argument("--validate-only",action="store_true",help="validate input and print bounded request metadata; no network call")
 args=parser.parse_args()
 try: payload=json.load(sys.stdin)
 except Exception: print(json.dumps({"status":"unknown","reason":"invalid_json","advisory_only":True}),file=sys.stdout); return 2
 try: clean=core.prepare_input(payload)
 except core.InputError as exc: print(json.dumps({"status":"unknown","reason":str(exc),"advisory_only":True},ensure_ascii=False)); return 2
 if args.validate_only:
  try: request=core.build_request(clean,model="jev-latest")
  except core.InputError as exc: print(json.dumps({"status":"unknown","reason":str(exc),"advisory_only":True},ensure_ascii=False)); return 2
  state=json.loads(request["state"])
  print(json.dumps({"status":"valid","taxonomy_version":core.TAXONOMY_VERSION,"segments":[{"id":r["id"],"start":r["start"],"end":r["end"],"offset_scope":r["offset_scope"],"skipped":r["skipped"]} for r in clean["segments"]],"request_segment_count":len(state["segments"]),"network_calls":0,"advisory_only":True},ensure_ascii=False))
  return 0
 print(plugin.ProseFeatureTool(Context())(payload))
 return 0
if __name__=="__main__": raise SystemExit(main())
