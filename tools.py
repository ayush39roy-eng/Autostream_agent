
# tools.py — Mock lead capture tool for AutoStream AI Agent.




def mock_lead_capture(name: str, email: str, platform: str) -> str:
    
    print(f"\n{'='*50}")
    print(f"📋 LEAD CAPTURED SUCCESSFULLY")
    print(f"{'='*50}")
    print(f"  Name:     {name}")
    print(f"  Email:    {email}")
    print(f"  Platform: {platform}")
    print(f"{'='*50}\n")
    
    return f"Lead captured successfully: {name}, {email}, {platform}"
