"""Metadata-only provider lifecycle logging; all requests use an offline mock."""
import importlib.util, json, logging, sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("jev_prose_lifecycle_test_plugin",ROOT/"__init__.py",submodule_search_locations=[str(ROOT)])
plugin=importlib.util.module_from_spec(spec); sys.modules[spec.name]=plugin; spec.loader.exec_module(plugin)
class Context:
 def get_config(self,key,default=None): return "typesafe" if key=="primary_provider" else default
def test_success_and_timeout_are_correlated_and_body_free():
 capture=[]
 class Handler(logging.Handler):
  def emit(self,record): capture.append(record.getMessage())
 logger=logging.getLogger("hermes_plugins.jev_lifecycle.prose"); handler=Handler(); logger.addHandler(handler)
 try:
  with patch.object(plugin,"_send",return_value={"answers":{}}):
   result=json.loads(plugin.ProseFeatureTool(Context())({"segments":[{"id":"x","text":"PRIVATE_SENTINEL"}]}))
  assert result["status"]=="unknown"  # invalid mock shape fails open, but transport succeeds
  events=[row.split("event=")[1].split()[0] for row in capture]
  assert events==["provider_start","transport_response","delivered"]
  assert "PRIVATE_SENTINEL" not in "\n".join(capture)
  capture.clear()
  with patch.object(plugin,"_send",side_effect=TimeoutError):
   result=json.loads(plugin.ProseFeatureTool(Context())({"segments":[{"id":"x","text":"offline"}]}))
  assert result["status"]=="unknown"
  assert [row.split("event=")[1].split()[0] for row in capture]==["provider_start","provider_failure","delivered"]
 finally: logger.removeHandler(handler)
