import pandas as pd
import numpy as np
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class MLBStatsAPI:
  base_url = "https://statsapi.mlb.com/api/v1"
  """
  Wrapper for the MLB Stats API
  """
  def __init__(self):
    self.session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500,502,503,504])
    self.session.mount('https://', HTTPAdapter(max_retries=retries))

  def get_season_schedule(self, season: int) -> pd.DataFrame:
        """
        Fetch schedule with hits/errors
        """
        url = f"{self.base_url}/schedule"

        params = {
            'sportId': 1,
            'season': season,
            'gameType': 'R',
            'hydrate': 'linescore'
        }

        print(f"Fetching schedule + stats for {season}...")
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"❌ API Error: {e}")
            return pd.DataFrame()

        games_list = []

        if 'dates' not in data:
            return pd.DataFrame()

        for date_obj in data['dates']:
            date = date_obj['date']
            for game in date_obj['games']:

                # 1. Get score data
                home_team_data = game['teams']['home']
                away_team_data = game['teams']['away']

                
                if 'score' not in home_team_data or 'score' not in away_team_data:
                    continue 

                # 2. Get 'linescore' stats
                linescore = game.get('linescore', {})
                ls_home = linescore.get('teams', {}).get('home', {})
                ls_away = linescore.get('teams', {}).get('away', {})


                games_list.append({
                    'game_id': game['gamePk'],
                    'date': date,
                    'home_team': home_team_data['team']['name'],
                    'away_team': away_team_data['team']['name'],
                    'home_score': home_team_data['score'], 
                    'away_score': away_team_data['score'], 
                    'home_hits': ls_home.get('hits', 0),   
                    'home_errors': ls_home.get('errors', 0),
                    'away_hits': ls_away.get('hits', 0),
                    'away_errors': ls_away.get('errors', 0)
                })

        return pd.DataFrame(games_list)


class BettingDataLoader:
  def __init__(self, filepath:str):
    self.filepath = filepath

  def load_odds(self, target_book: str = 'bet365') -> pd.DataFrame:
    print(f"Loading odds from {self.filepath} using {target_book}...")

    if not os.path.exists(self.filepath):
      raise FileNotFoundError(f"Could not find file: {self.filepath}")

    with open(self.filepath, 'r') as f:
      raw_data = json.load(f)

    if isinstance(raw_data, list):
      return pd.json_normalize(raw_data, sep='_')

    elif isinstance(raw_data, dict):
      games_list = []

      for date, games in raw_data.items():
          for game in games:
              view = game.get('gameView', {})

              # Filter for Regular Season games
              g_type = view.get('gameType', 'R')
              if g_type != 'R':
                  continue

              
              odds_section = game.get('odds', {})
              moneylines = odds_section.get('moneyline', [])

              selected_book = None

              # 1. Try Target (Case Insensitive)
              for book in moneylines:
                  if book.get('sportsbook', '').lower() == target_book.lower():
                      selected_book = book
                      break

              # 2. Try Fallbacks (Case Insensitive)
              if selected_book is None and moneylines:
                  fallbacks = ['pinnacle', 'caesars', 'draftkings', 'fanduel']
                  for fb in fallbacks:
                      for book in moneylines:
                          # FIX: Added .lower() here too
                          if book.get('sportsbook', '').lower() == fb:
                              selected_book = book
                              break
                      if selected_book: break

                  # 3. Last Resort: Take the first one
                  if selected_book is None:
                      selected_book = moneylines[0]

              game_info = {
                  'date': date,
                  'away_team_abbr': view.get('awayTeam', {}).get('shortName'),
                  'away_score': view.get('awayTeamScore'),
                  'home_team_abbr': view.get('homeTeam', {}).get('shortName'),
                  'home_score': view.get('homeTeamScore'),
                  'game_type': g_type # Saved for audit
              }

              if selected_book:
                  current = selected_book.get('currentLine', {})
                  game_info['home_moneyline'] = current.get('homeOdds')
                  game_info['away_moneyline'] = current.get('awayOdds')
                  game_info['sportsbook'] = selected_book.get('sportsbook')
              else:
                  game_info['home_moneyline'] = np.nan
                  game_info['away_moneyline'] = np.nan
                  game_info['sportsbook'] = None

          
              games_list.append(game_info)

      return pd.json_normalize(games_list, sep='_')


def load_and_merge_data(
