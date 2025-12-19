from datetime import datetime
from zoneinfo import ZoneInfo

def get_dc_time():
    # Get the timezone for Washington, DC (US/Eastern)
    dc_tz = ZoneInfo('America/New_York')
    
    # Get current time in DC
    dc_time = datetime.now(dc_tz)
    
    # Format and return the time
    return dc_time.strftime('%Y-%m-%d %I:%M:%S %p %Z')

if __name__ == "__main__":
    print(f"The current time in Washington, DC is: {get_dc_time()}")
    print("Note: This time automatically adjusts for DST when applicable.")
