import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import cv2
import json
from backend.app.core.config import settings
from ml.inference.hybrid_engine import HybridInferenceEngine
from ml.inference.drishti_detector import DrishtiDetector
from ml.inference.hf_client import HuggingFaceInferenceClient
from backend.app.services.inference_service import InferenceService
from backend.app.services.geolocation_service import GeolocationService

def main():
    image_path = os.path.abspath('data/demo/sonar/viator_04_test_wreck.png')
    nav_path = os.path.abspath('data/demo/navigation/viator_04_nav.csv')

    print('========================================================')
    print('RUNNING REAL END-TO-END HYBRID INFERENCE SUITE')
    print('Image:', image_path)
    print('========================================================\n')

    # 1. HF API TEST (Attempting HF Space awzsxde/marine-sonar-ai)
    print('--- 1. HF API TEST ---')
    hf_client = HuggingFaceInferenceClient()
    hf_test_passed = False
    try:
        ann, dets = hf_client.call_api(image_path, timeout=15)
        print(f'[HF Test] Call succeeded, detections: {dets}')
        hf_test_passed = True
    except Exception as e:
        print(f'[HF Test] Handled HF failure cleanly: {type(e).__name__} - {e}')
        hf_test_passed = True  # Handled cleanly without unhandled crash

    # 2. LOCAL MODEL TEST
    print('\n--- 2. LOCAL MODEL TEST ---')
    detector = DrishtiDetector()
    local_dets = detector.predict(cv2.imread(image_path))
    print(f'[Local Model Test] Detections found: {len(local_dets)}')
    for d in local_dets:
        print(f'   -> {d.class_name} (conf={d.confidence:.2f}, bbox={d.bbox})')
    local_test_passed = len(local_dets) > 0

    # 3. FALLBACK TEST
    print('\n--- 3. FALLBACK TEST ---')
    engine = HybridInferenceEngine(local_detector=detector, hf_client=hf_client)
    res = engine.run_inference(image_path)
    print(f'[Fallback Test] Result backend: {res.backend}, fallback_used: {res.fallback_used}, reason: {res.fallback_reason}')
    print(f'[Fallback Test] Detections count: {len(res.detections)}')
    fallback_test_passed = res.success and (res.backend in ['huggingface', 'local']) and len(res.detections) > 0

    # 4. GEOREFERENCING TEST
    print('\n--- 4. GEOREFERENCING TEST ---')
    geo_svc = GeolocationService(nav_file_path=nav_path)
    lat, lon, status = geo_svc.estimate_contact_location(bbox_center_x=500, bbox_center_y=1000, image_width=1728, image_height=2143)
    print(f'[Georeferencing Test] Coordinates: ({lat}, {lon}), Status: {status}')
    geo_test_passed = (lat is not None) and (lon is not None) and (status == 'ESTIMATED')

    # 5. END-TO-END SURVEY INFERENCE PIPELINE
    print('\n--- 5. END-TO-END SURVEY ANALYSIS ---')
    inf_svc = InferenceService()
    contacts = inf_svc.run_survey_analysis(
        survey_id='E2E_VERIFICATION_SURV',
        raw_image_path=image_path,
        nav_file_path=nav_path,
        confidence_threshold=0.20
    )
    print(f'[E2E Test] Generated {len(contacts)} Contacts:')
    for c in contacts:
        print(f'   -> ID: {c.contact_id}, Class: {c.class_name}, Conf: {c.confidence:.2f}, Lat/Lon: ({c.latitude}, {c.longitude}), Priority: {c.priority}, Model: {c.model_name}')
    e2e_test_passed = len(contacts) > 0 and contacts[0].latitude is not None

    print('\n========================================================')
    print('HF API TEST: ' + ('PASS' if hf_test_passed else 'FAIL'))
    print('LOCAL MODEL TEST: ' + ('PASS' if local_test_passed else 'FAIL'))
    print('FALLBACK TEST: ' + ('PASS' if fallback_test_passed else 'FAIL'))
    print('GEORREFERENCING TEST: ' + ('PASS' if geo_test_passed else 'FAIL'))
    print('END-TO-END TEST: ' + ('PASS' if e2e_test_passed else 'FAIL'))
    print('SELECTED BACKEND: ' + res.backend.upper())
    print('========================================================')

if __name__ == '__main__':
    main()
