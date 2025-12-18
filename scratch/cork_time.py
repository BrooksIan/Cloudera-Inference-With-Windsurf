import pytz
from datetime import datetime

def get_cork_time():
    """Get the current time in Cork, Ireland."""
    # Define the timezone for Cork, Ireland (Europe/Dublin is the correct timezone for Ireland)
    cork_tz = pytz.timezone('Europe/Dublin')
    
    # Get current time in UTC
    utc_now = datetime.now(pytz.utc)
    
    # Convert to Cork time
    cork_time = utc_now.astimezone(cork_tz)
    
    # Format the time
    time_format = "%Y-%m-%d %H:%M:%S %Z%z"
    return cork_time.strftime(time_format)

if __name__ == "__main__":
    try:
        current_time = get_cork_time()
        print(f"Current time in Cork, Ireland: {current_time}")
    except Exception as e:
        print(f"Error getting the time: {e}")
