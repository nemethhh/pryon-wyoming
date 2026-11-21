"""Oracle output parser and runner argument checks.

REAL_ORACLE_OUTPUT is a captured pryon_test run (log text only): stdout first,
then stderr, as run_oracle concatenates them.
"""
import json
from pathlib import Path

import pytest

from pryon_oracle import OracleError, parse_detections, run_oracle

REAL_ORACLE_OUTPUT = r"""attributes: {"object_type_name":"pryon_library_attributes","object_type_version":1,"pryon_library_engine_compatibility_info":{"aed_ecids":[1,2,3,5,6,7],"enrollment_profile_current_ecid":2,"enrollment_profile_earliest_supported_ecid":1,"fingerprint_ecids":[1,2,3,4],"interaction_context_sdsm_current_ecid":1,"interaction_context_sdsm_earliest_supported_ecid":1,"mutable_model_current_ecid":5,"mutable_model_earliest_supported_ecid":5,"ntt_dvad_ecids":[1,2,3,4,5,6,7,8,9,10],"pryon_engine_uuid":"da5620af-cace-58d0-ae0d-39724cc0b2a8","speaker_id_model_current_ecid":3,"speaker_id_model_earliest_supported_ecid":3,"supplemental_model_current_ecid":5,"supplemental_model_earliest_supported_ecid":4,"version":1,"wakeword_ecids":[1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]},"pryon_library_version_info":{"atlas":"","build_type":"Release","compiler":"Clang 12.0.9","git_version":"UNKNOWN-2.41-gUNKNOWN","host_platform":"x86_64-Linux-4.14.355-280.710.amzn2.x86_64","target_platform":"armv7-a-Android-1","uuid":"da5620af-cace-58d0-ae0d-39724cc0b2a8","version":2}}
format: enc=0 rate=16000 bits=16 ch=1 type0=0
pryon[4] INFO: The log label 'KWS' should be defined before calling AssignThreadLoggerByLabel!
RESULT decoder=dec keyword=ALEXA type=2 begin=22240 end=32640 (1.39s..2.04s) meta=1576
META header 24 bytes, {"activeKeywords":["ALEXA","STOP"],"clientProperties":{},"cmsSampleIndex":14240,"cmsWindowCount":304522112,"cmsWindowNumFrames":304522112,"cmsWindowSumOfSquares":" [ 3.5964e+10 3.81024e+10 3.91397e+10 4.13165e+10 4.27415e+10 4.22579e+10 4.17278e+10 4.23712e+10 4.38455e+10 4.4754e+10 4.48786e+10 4.40884e+10 4.41784e+10 4.45015e+10 4.33838e+10 4.35009e+10 4.36705e+10 4.32117e+10 4.34785e+10 4.50926e+10 ]\n","cmsWindowSums":" [ 3.2284e+09 3.31208e+09 3.34733e+09 3.43886e+09 3.50723e+09 3.4984e+09 3.48506e+09 3.51419e+09 3.5751e+09 3.6142e+09 3.62405e+09 3.59189e+09 3.58767e+09 3.59391e+09 3.54618e+09 3.54663e+09 3.55304e+09 3.53926e+09 3.55707e+09 3.63404e+09 ]\n","deviceMetrics":{"adjusted_kw_indices":0,"audio_backlog_at_result_building_time_msec":2095,"decoderAcceptCount":1,"decoderNMCount":0,"decoderUpTimeMsec":1083,"erle_suppression_decision":0,"fl_fr_count":0,"fl_init_state":0,"fp_ignored_strong_match":0,"has_audio_lsb_metadata":0,"is_first_decoder_ww_event":1,"kws_appraise_candidate_msec":0,"last_seen_device_mute_sample":0,"last_seen_digital_silence_sample":29360,"max_audio_backlog_msec":8350,"max_rss_delta_kb":7408,"max_rss_kb":64212,"nm_since_last_ww_count":0,"num_digital_silence_frames_since_last_ww":154,"num_of_acoustic_events_since_last_ww":0,"percentage_of_frames_evaluated_for_acoustic_event_since_last_ww":0,"prev_erle_count":0,"prev_strong_fp_count":0,"prev_strong_wm_count":0,"pryon_cpu_time_process_msec":1071,"pryon_cpu_time_thread_msec":1063,"pryon_latency_msec":7403,"pryon_latency_msec_old":1112,"pryon_model_latency_msec":70,"pryon_old_upl_msec":7473,"skipped_audio_backlog_since_last_ww_msec":0,"sum_client_sample_index_gaps_since_last_ww":0,"time_since_last_nm_msec":1970,"time_since_last_ww_msec":1970,"wakeword_len_msec":580,"wm_ignored_strong_match":0},"kwActiveThreshold":0.923000,"kwClassificationScore":0.936928,"kwDetectionType":"Accept","kwHmmScore":-340282346638528859811704183484516925440,"kwName":"ALEXA","kwSampleDetectionIndex":32640,"kwSampleEndIndex":31520,"kwSampleStartIndex":22240,"kwStats":{"kw-stats":[{"deduped_statistics":{"classification":{"histogram":{"counts":[2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"max-value":0.998279,"min-value":-0.001721}},"segmentation":{"histogram":{"counts":[0],"max-value":1.440944,"min-value":0.440944}}},"journal":{"events":[],"version":1},"name":"ALEXA"}],"version":1},"nttModeChangesAndDetects":[0,0,0,0,0,0],"object_type_name":"pryon_lookahead_metadata","object_type_version":3,"pastStrongFps":[],"pryonBuildUUID":"da5620af-cace-58d0-ae0d-39724cc0b2a8","pryonVersion":"UNKNOWN-2.41-gUNKNOWN","streamId":"1795288600821090WWl0wbNuBCuEoO","watermarkMatchInfo":{"decodedKeyHash":"31daa0390e909cfa49d37b2cd437a18b","sampleIdx":32800,"snrScore":0.977116,"snrThreshold":5.080000,"wmDetectionsSinceLastAccept":0,"wmScore":7.309614,"wmStrongMatch":false},"wwModelChecksum":"74a26372ecb79fc6e5954d9f9fff29a3"}
RESULT decoder=dec keyword=ALEXA type=0 begin=61760 end=89920 (3.86s..5.62s) meta=1512
META header 24 bytes, {"activeKeywords":["ALEXA"],"clientProperties":{},"deviceMetrics":{"adjusted_kw_indices":0,"audio_backlog_at_result_building_time_msec":5565,"decoderAcceptCount":1,"decoderNMCount":1,"decoderUpTimeMsec":2831,"erle_suppression_decision":0,"fl_fr_count":0,"fl_init_state":0,"fp_ignored_strong_match":0,"has_audio_lsb_metadata":0,"is_first_decoder_ww_event":0,"kws_appraise_candidate_msec":0,"last_seen_device_mute_sample":0,"last_seen_digital_silence_sample":80400,"max_audio_backlog_msec":0,"max_rss_delta_kb":256,"max_rss_kb":64468,"nm_since_last_ww_count":1,"num_digital_silence_frames_since_last_ww":156,"num_of_acoustic_events_since_last_ww":0,"percentage_of_frames_evaluated_for_acoustic_event_since_last_ww":0,"prev_erle_count":0,"prev_strong_fp_count":0,"prev_strong_wm_count":0,"pryon_cpu_time_process_msec":2796,"pryon_cpu_time_thread_msec":2791,"pryon_latency_msec":5574,"pryon_latency_msec_old":2817,"pryon_model_latency_msec":1150,"pryon_old_upl_msec":6724,"skipped_audio_backlog_since_last_ww_msec":0,"sum_client_sample_index_gaps_since_last_ww":0,"time_since_last_nm_msec":4470,"time_since_last_ww_msec":2500,"wakeword_len_msec":610,"wm_ignored_strong_match":0},"kwActiveThreshold":0.690000,"kwClassificationScore":0.913608,"kwDetectionType":"NearMiss","kwHmmScore":-340282346638528859811704183484516925440,"kwName":"ALEXA","kwSampleDetectionIndex":89920,"kwSampleEndIndex":71520,"kwSampleStartIndex":61760,"kwStats":{"kw-stats":[{"deduped_statistics":{"classification":{"histogram":{"counts":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],"max-value":0.998279,"min-value":-0.001721}},"segmentation":{"histogram":{"counts":[0],"max-value":1.440944,"min-value":0.440944}}},"journal":{"events":[[2470,0,0.936928,0.923000,580,0]],"version":1},"name":"ALEXA"}],"version":1},"nttModeChangesAndDetects":[0,0,0,0,0,0],"object_type_name":"pryon_lookahead_metadata","object_type_version":3,"pastStrongFps":[],"pryonBuildUUID":"da5620af-cace-58d0-ae0d-39724cc0b2a8","pryonVersion":"UNKNOWN-2.41-gUNKNOWN","streamId":"1795288602569413WW01WA+OfBw9dv","wwModelChecksum":"74a26372ecb79fc6e5954d9f9fff29a3"}
pryon[4] INFO: DecoderID=dec: recognition_id=dec-0: {KWS} Endpointing decisions are [ all_speech ].
pryon[4] INFO: DecoderID=dec: recognition_id=dec-0: {KWS} Observed consecutive frames of digital silence: 149
pryon[4] INFO: DecoderID=dec: recognition_id=dec-0: {KWS} segmentation_score=-inf,classifier_score=0.936928,active_threshold=0.923,detection_type=Accept
pryon[4] INFO: DecoderID=dec: recognition_id=dec-1: {KWS} segmentation_score=-inf,classifier_score=0.913608,active_threshold=0.69,detection_type=NearMiss
pushed 134407 samples (8.4s), 2 callbacks
"""


EXPECTED_ALEXA_DETECTIONS = [
    {
        "keyword": "ALEXA",
        "detection_type": "Accept",
        "classifier_score": 0.936928,
        "begin_sample": 22240,
        "end_sample": 32640,
        "active_threshold": 0.923,
    },
    {
        "keyword": "ALEXA",
        "detection_type": "NearMiss",
        "classifier_score": 0.913608,
        "begin_sample": 61760,
        "end_sample": 89920,
        "active_threshold": 0.69,
    },
]


def test_parse_real_oracle_output() -> None:
    assert parse_detections(REAL_ORACLE_OUTPUT) == EXPECTED_ALEXA_DETECTIONS


def test_parse_output_with_no_detections() -> None:
    quiet = (
        'attributes: {"object_type_name":"pryon_library_attributes"}\n'
        "format: enc=0 rate=16000 bits=16 ch=1 type0=0\n"
        "pushed 134407 samples (8.4s), 0 callbacks\n"
    )
    assert parse_detections(quiet) == []


def test_parse_result_without_kws_line_raises() -> None:
    out = "RESULT decoder=dec keyword=ALEXA type=2 begin=22240 end=32640 (1.39s..2.04s) meta=1576\n"
    with pytest.raises(OracleError, match="out of sync"):
        parse_detections(out)


def test_parse_type_contradiction_raises() -> None:
    # RESULT says type=2 (Accept) but the paired KWS log line says NearMiss.
    out = (
        "RESULT decoder=dec keyword=ALEXA type=2 begin=22240 end=32640 (1.39s..2.04s) meta=1576\n"
        "pryon[4] INFO: DecoderID=dec: recognition_id=dec-0: {KWS} segmentation_score=-inf,"
        "classifier_score=0.5,active_threshold=0.923,detection_type=NearMiss\n"
    )
    with pytest.raises(OracleError, match="contradicts"):
        parse_detections(out)


def test_alexa_espeak_fixture_anchor_values() -> None:
    """The committed fixture (a fresh oracle run on the espeak clip)."""
    fixture = Path(__file__).parent / "fixtures" / "alexa_espeak_oracle.json"
    assert json.loads(fixture.read_text()) == EXPECTED_ALEXA_DETECTIONS


def test_run_oracle_missing_clip_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="audio clip not found"):
        run_oracle(tmp_path / "nope.raw")


def test_run_oracle_missing_binary_raises(tmp_path, monkeypatch) -> None:
    import pryon_oracle as oracle

    clip = tmp_path / "clip.raw"
    clip.write_bytes(b"\x00\x00")
    monkeypatch.setattr(oracle, "ORACLE_BIN", tmp_path / "pryon_test")
    with pytest.raises(OracleError, match="tools/oracle/build.sh"):
        oracle.run_oracle(clip)


def test_run_oracle_missing_rootfs_raises(tmp_path, monkeypatch) -> None:
    import shutil

    import pryon_oracle as oracle

    clip = tmp_path / "clip.raw"
    clip.write_bytes(b"\x00\x00")
    monkeypatch.setattr(oracle, "ORACLE_BIN", clip)  # exists -> check passes
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("PYPRYON_ORACLE_ROOTFS", str(tmp_path / "missing"))
    with pytest.raises(OracleError, match="PYPRYON_ORACLE_ROOTFS"):
        oracle.run_oracle(clip)


def test_run_oracle_rejects_bad_client_props(tmp_path: Path) -> None:
    with pytest.raises(OracleError, match="must be an int"):
        run_oracle(tmp_path / "clip.raw", client_props={"AlarmState": 1.5})
    nine = {f"P{i}": 1 for i in range(9)}
    with pytest.raises(OracleError, match="at most 8"):
        run_oracle(tmp_path / "clip.raw", client_props=nine)
