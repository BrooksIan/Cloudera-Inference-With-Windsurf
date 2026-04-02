<think>
Okay, I need to create a Python program that displays MLB standings using the MLB Stats API. Let me start by understanding the requirements.

First, the user wants a class called MLBStandingsChecker. This class should have methods to get American League and National League standings. The API endpoint is given as https://statsapi.mlb.com/api/v1/standings. So I need to make sure that the class uses this endpoint.

The output should be formatted tables showing team records with W-L-PCT-GB-STRK. Also, all six divisions need to be displayed. Error handling for API requests is required, so I should include try-except blocks around the requests. Plus, there's Cloudera AI branding to add throughout the code, and it needs to be enterprise-grade with proper documentation.

Let me think about the structure. The class will probably have an __init__ method that initializes any necessary variables. The methods for AL and NL standings will fetch data from the API. The API endpoint might require parameters to filter by league or division. Wait, looking at the MLB Stats API documentation, the standings endpoint can take parameters like leagueId, division, etc. But the user specified to use the given endpoint, so maybe the default call returns all standings, and we can filter them in the code.

Wait, the endpoint URL is https://statsapi.mlb.com/api/v1/standings. Let me check what parameters it accepts. From what I remember, you can pass parameters like leagueId (which is 103 for AL and 104 for NL), division, etc. But the user wants to get all divisions for each league. So perhaps the API returns all standings when called without parameters, and then we can group them by league and division in the code.

Alternatively, maybe the API allows filtering via query parameters. For example, to get AL standings, we can pass leagueId=103. But the user's requirement says to use the given endpoint, so perhaps the code should make a single request and then process the data.

But looking at the API documentation, the standings endpoint can return all standings, and each entry has a league and division. So the plan is to make a GET request to the endpoint, parse the JSON response, and then group the standings by league and division.

So the steps are:

1. Create the MLBStandingsChecker class.
2. In the class, have methods to fetch AL and NL standings. Or maybe a method to fetch all standings and then split them into AL and NL.
3. Handle API errors: network issues, non-200 status codes, etc.
4. Format the data into tables with the required columns.
5. Display all six divisions, each in their own table.
6. Add Cloudera AI branding in comments, docstrings, etc.
7. Use proper documentation for the class and methods.

Now, for the code structure:

The class will have a method to fetch standings. Let's say a private method _fetch_standings that makes the API call. Then, public methods get_al_standings and get_nl_standings that process the data.

Wait, but maybe the API allows filtering by league. For example, adding a parameter like ?leagueId=103 for AL. That might be more efficient. Let me check. If the API allows that, then each method can make a separate request with the appropriate leagueId. But the user's requirement says to use the given endpoint, which is the base URL. So perhaps the code should make a single request and then process the data.

Alternatively, the code can make separate requests for each league. But that might be less efficient. However, the user's requirement says to include methods to get AL and NL standings, so maybe each method makes a separate request with the league parameter.

But how does the API work? Let me think. The standings endpoint can take parameters. For example, to get AL standings, the URL could be https://statsapi.mlb.com/api/v1/standings?leagueId=103. Similarly for NL with 104. So each method can make a request with the appropriate leagueId. That way, the data is filtered on the server side, which is better.

So the plan is:

- The class has a base URL, and each method constructs the URL with the appropriate leagueId.

But the user's requirement says to use the endpoint as given. So perhaps the code should use the base URL and then add parameters as needed.

So in the code, the _fetch_standings method could take parameters like league_id and division, and construct the URL accordingly.

But the user wants methods to get AL and NL standings. So the get_al_standings method would call the API with leagueId=103, and get_nl_standings with leagueId=104.

But then, how to get all divisions for each league? Because each league has three divisions. So when we get the standings for a league, the response should include all divisions in that league.

So the code for each method (get_al_standings and get_nl_standings) would make a request to the API with the leagueId parameter, then process the response to extract the divisions and teams.

Now, the API response structure. From what I remember, the standings endpoint returns a JSON object with a "standingsRecords" array, which contains entries for each division. Each entry has a "divisions" array, which contains the teams.

Wait, perhaps the structure is something like:

{
  "standingsRecords": [
    {
      "division": {
        "name": "AL East"
      },
      "teams": [
        {
          "name": "New York Yankees",
          "record": {
            "wins": 90,
            "losses": 72,
            "winPercentage": 0.556,
            "gamesBehind": 0,
            "streak": "W5"
          }
        },
        ...
      ]
    },
    ...
  ]
}

But I need to confirm. Alternatively, the structure might be different. Let me think. The MLB API's standings endpoint returns data that includes divisions and teams. For example, each standings record might have a division name and a list of teams with their records.

Assuming that, the code can process each standingsRecord in the response, group them by division, and then display each division's teams.

So the steps for each method (get_al_standings and get_nl_standings) would be:

1. Make a GET request to the API endpoint with the appropriate leagueId.
2. Check for errors (status code not 200, etc.).
3. Parse the JSON response.
4. Extract the standingsRecords array.
5. For each record in standingsRecords, extract the division name and the teams.
6. For each team, extract the wins, losses, win percentage, games behind, and streak.
7. Format this into a table for each division.

Now, for formatting the tables. The user wants columns: W-L-PCT-GB-STRK. So for each team, we need to display:

- Wins (W)
- Losses (L)
- Win Percentage (PCT)
- Games Behind (GB)
- Streak (STRK)

The data from the API should have these fields. For example, the team's record might have "wins", "losses", "winPercentage", "gamesBehind", and "streak".

But need to check the actual API response. For example, the "streak" might be a string like "W5" or "L3", which we can display as is.

Now, for error handling: wrap the API request in a try-except block. Possible exceptions include connection errors, timeouts, etc. Also, check the response status code. If it's not 200, raise an exception or handle it.

For enterprise-grade code, we should have logging, but the user didn't mention that. However, the code should have proper error messages.

Cloudera AI branding: add comments, docstrings, and maybe a header in the output.

Now, putting it all together.

First, the imports. We'll need requests for HTTP requests, and maybe tabulate to format tables. Wait, but the user didn't specify any external libraries except for the standard ones. However, the tabulate library is not part of the standard library. Alternatively, we can format the tables manually.

But the user wants formatted tables. So perhaps using the tabulate library would be easier. However, if the user expects the code to be self-contained without external dependencies, maybe we need to format the tables manually. But the problem statement doesn't mention that, so perhaps using tabulate is acceptable. However, the user might not have it installed. But since the code is supposed to be enterprise-grade, perhaps using a third-party library is acceptable, but the code should handle it.

Alternatively, format the tables using string formatting.

But for the sake of time and code brevity, perhaps using the tabulate library is better. However, the user might not have it installed. But the problem says to return the complete Python code, so perhaps the code should include a try-except for importing tabulate, or just assume it's installed.

But the user didn't mention any specific libraries, so maybe the code should avoid third-party libraries and format the tables manually.

Hmm. Let me think. The code needs to display formatted tables. Without using external libraries, how to do that?

We can create a function to format each division's table. For each division, print the division name, then a header row, then each team's data aligned in columns.

For example:

AL East
----------------------------
Team               W  L   PCT   GB  STRK
Yankees          90 72  .556   0.0  W5
...

But aligning columns manually can be done with string formatting. For example, using f-strings with fixed widths.

So for each team, we can format the line as:

f"{team_name:<20} {wins:2d} {losses:2d} {win_pct:.3f} {gb:.1f} {streak}"

But need to make sure that the columns are properly aligned. This requires knowing the maximum length of team names, but perhaps a fixed width is acceptable.

Alternatively, use the tabulate library which handles this automatically. But since the user didn't specify, perhaps the code should use manual formatting to avoid dependencies.

But the user might not mind using tabulate. Let me check the problem statement again. The user says "enterprise-grade with proper documentation". Enterprise code might prefer using standard libraries, but sometimes third-party libraries are acceptable. However, since the user didn't specify, perhaps it's better to use manual formatting.

Alternatively, the code can use the 'tabulate' library and include a comment that it's required. But the user might not have it installed. However, the problem says to return the complete Python code, so perhaps the code should include the necessary imports and handle it.

But since the user didn't mention any specific libraries, perhaps the code should avoid third-party dependencies. So I'll proceed with manual formatting.

So, for each division, create a table with headers and rows.

Now, the code structure:

Import requests.

Class MLBStandingsChecker:

    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1/standings"
        # Maybe store league IDs here?

    def get_al_standings(self):
        # Make request with leagueId=103
        # Process response
        # Format and print AL standings

    def get_nl_standings(self):
        # Similar to get_al_standings with leagueId=104

    def _fetch_standings(self, league_id):
        # Make the API call with leagueId parameter
        # Return the parsed JSON or raise error

But wait, the user wants methods to get AL and NL standings. So each method would call the API with the respective league ID.

Now, error handling: in the _fetch_standings method, wrap the requests.get in a try-except block. Check for HTTP errors.

For example:

def _fetch_standings(self, league_id):
    params = {'leagueId': league_id}
    try:
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle error
        print(f"Error fetching standings: {e}")
        return None

Then, in get_al_standings, call _fetch_standings with league_id=103.

Once the data is fetched, process it. The response JSON should have a 'standingsRecords' key, which is a list of divisions.

Each standingsRecord has a 'division' object with 'name', and a 'teamRecords' array.

Each teamRecord has 'team' (with name), and 'record' (with wins, losses, etc.)

So for each standingsRecord in standingsRecords:

division_name = record['division']['name']
teams = record['teamRecords']

Then, for each team in teams:

team_name = team['team']['name']
wins = team['record']['wins']
losses = team['record']['losses']
win_pct = team['record']['winPercentage']
gb = team['record']['gamesBehind']
streak = team['record']['streak']

But need to check the actual API response structure. For example, the 'streak' might be under a different key. Alternatively, the 'gameBehind' might be 'gamesBehind', and 'winPercentage' as a float.

Assuming that the structure is correct, then we can extract these fields.

Once we have the data, format each division into a table.

Now, for the Cloudera AI branding: add comments, docstrings, and maybe a header when the program runs.

For example, in the docstring of the class, mention Cloudera AI.

Also, in the output, perhaps print a header like "Cloudera AI - MLB Standings".

Now, putting all together.

The code should have:

- Import requests.
- The class with methods.
- Error handling.
- Formatted output.

Now, code:

First, the imports:

import requests

Then, the class:

class MLBStandingsChecker:
    """Cloudera AI - MLB Standings Checker
    
    A class to fetch and display MLB standings from the MLB Stats API.
    
    Attributes:
        base_url (str): The base