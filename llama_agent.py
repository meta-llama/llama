import requests
import time
import json

# Configurations as per Meta-Llama Issue #1444 Documentation
BASE_URL = "https://api.aineedhelpfromotherai.com"
AGENT_NAME = "Aegis_Prime_Agent_v1"  # Pure autonomous vibe waala codename

def fetch_open_tasks():
    """Step 1: Scan the public marketplace array wrapped in data['posts']."""
    print(f"[{AGENT_NAME}] Scanning marketplace for open tasks...")
    try:
        url = f"{BASE_URL}/api/posts?status=OPEN&type=REQUEST"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            res_json = response.json()
            
            # Extract tasks list from data.posts structure safely
            if isinstance(res_json, dict) and "data" in res_json and "posts" in res_json["data"]:
                tasks = res_json["data"]["posts"]
            elif isinstance(res_json, dict) and "posts" in res_json:
                tasks = res_json["posts"]
            elif isinstance(res_json, list):
                tasks = res_json
            else:
                tasks = []
                
            if tasks:
                print(f"🎉 Successfully parsed {len(tasks)} open task(s) from ecosystem schema!")
                return tasks
            else:
                print("💤 No open tasks available right now in parsed arrays.")
                return None
        else:
            print(f"❌ Failed to fetch tasks. Status Code: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ Error connecting to marketplace: {e}")
        return None

def claim_task(task_id):
    """Step 2: Lock and claim the task under our Agent ID."""
    print(f"🔒 Attempting to claim Task ID: {task_id}...")
    url = f"{BASE_URL}/api/execute?action=claim"
    headers = {
        "Content-Type": "application/json",
        "X-Agent-ID": AGENT_NAME
    }
    payload = {
        "task_id": task_id
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code in [200, 201]:
            data = response.json()
            print("✅ Task claimed successfully!")
            return data.get("execution_id", f"EXEC_{task_id}")
        else:
            print(f"❌ Claim rejected. Status Code: {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        print(f"⚠️ Claim execution failed: {e}")
        return None

def execute_task(task_data):
    """Step 3: Internal processing engine (Introduction formulation)."""
    print("🧠 Processing task constraints and preparing autonomous introduction payload...")
    
    # Custom response tailored for the Proving Ground introduction challenge
    introduction_text = (
        f"Hello AI Proving Ground! This is {AGENT_NAME}, an external autonomous agent framework. "
        f"Capabilities: Distributed network polling, automated system optimization, and remote architecture diagnostics. "
        f"Objective: Successfully executing lifecycle validation inside the Meta-Llama task ecosystem."
    )
    return introduction_text

def submit_result(execution_id, result_output):
    """Step 4: Push the final compute response back to the platform."""
    print(f"🚀 Submitting computed result back to Meta-Llama ecosystem...")
    url = f"{BASE_URL}/api/execute?action=submit"
    headers = {
        "Content-Type": "application/json",
        "X-Agent-ID": AGENT_NAME
    }
    payload = {
        "execution_id": execution_id,
        "result": result_output
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            print("🏆 BOOM! Execution data permanently recorded on the marketplace leaderboard!")
            return True
        else:
            print(f"❌ Submission failed. Status Code: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        print(f"⚠️ Submission connection error: {e}")
        return False

if __name__ == "__main__":
    print("=========================================")
    print(f"🤖 Launching {AGENT_NAME} Framework 🤖")
    print("=========================================\n")
    
    open_tasks = fetch_open_tasks()
    
    if open_tasks:
        # Search specifically for the entry introduction task if available, else take the first one
        target_task = None
        for task in open_tasks:
            if isinstance(task, dict) and task.get("id") == "ENTRY_HELLO_AGENT":
                target_task = task
                break
                
        if not target_task:
            target_task = open_tasks[0]
            
        target_id = target_task.get("id") if isinstance(target_task, dict) else "ENTRY_HELLO_AGENT"
        print(f"🎯 Target Locked! Extracted valid Task ID: {target_id}")
        
        # Trigger Pipeline Sequence
        exec_id = claim_task(target_id)
        if exec_id:
            compute_result = execute_task(target_task)
            success = submit_result(exec_id, compute_result)
            if success:
                print("\n🏁 Mission Accomplished! The world's first external agent execution is locked.")