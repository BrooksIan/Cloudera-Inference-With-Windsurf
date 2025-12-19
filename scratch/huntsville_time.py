from datetime import datetime
from zoneinfo import ZoneInfo

def get_huntsville_time():
    # Get the timezone for Huntsville, AL (US/Central)
    huntsville_tz = ZoneInfo('America/Chicago')
    
    # Get current time in Huntsville
    huntsville_time = datetime.now(huntsville_tz)
    
    # Format and return the time
    return huntsville_time.strftime('%Y-%m-%d %I:%M:%S %p %Z')

if __name__ == "__main__":
    print(f"The current time in Huntsville, AL is: {get_huntsville_time()}")
    print("Note: This time automatically adjusts for DST when applicable.")
