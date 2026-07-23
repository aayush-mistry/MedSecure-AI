import asyncio
import sys
from main import run_full_pipeline

async def test_pipeline(image_path):
    print(f"\n======================================")
    print(f"Testing Pipeline on: {image_path}")
    print(f"======================================")
    
    # Run the pipeline synchronously for testing
    from main import scan_progress_store
    run_full_pipeline("test-scan", image_path)
    
    if "test-scan" not in scan_progress_store or "result" not in scan_progress_store["test-scan"]:
        print("Pipeline failed to produce a result.")
        if "error" in scan_progress_store.get("test-scan", {}):
            print(f"Error: {scan_progress_store['test-scan']['error']}")
        return
        
    result = scan_progress_store["test-scan"]["result"]
    
    print("\n--- Final Authenticity Result ---")
    print(f"Overall Score: {result['authenticity_score']}")
    print(f"Verdict: {result['verdict']}")
    print(f"Explanation: {result['explanation']}")
    
    print("\n--- Detailed Results ---")
    for field, res in result['db_match_results'].items():
        if isinstance(res, dict):
            status = res.get('status', 'Unknown')
            val = res.get('extracted', 'None')
            print(f"  {field.ljust(20)}: {status.ljust(15)} | Extracted: {str(val).ljust(20)}")
            
    print("======================================\n")

if __name__ == "__main__":
    test_images = [
        "..\\frontend\\public\\samples\\calpol_genuine.jpg",
        "..\\frontend\\public\\samples\\crocin_counterfeit.jpg",
        "..\\frontend\\public\\samples\\omez_counterfeit.jpg"
    ]
    if len(sys.argv) > 1:
        test_images = [sys.argv[1]]
        
    for img in test_images:
        try:
            asyncio.run(test_pipeline(img))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Test failed for {img}: {e}")
