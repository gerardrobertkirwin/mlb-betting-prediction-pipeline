import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

TEAM_MAPPING = {
        # East
        'New York Yankees': 'NYY', 'NY Yankees': 'NYY', 'New York (AL)': 'NYY',
        'Boston Red Sox': 'BOS', 'Boston': 'BOS',
        'Tampa Bay Rays': 'TB', 'Tampa Bay': 'TB', 'Tampa': 'TB',
        'Toronto Blue Jays': 'TOR', 'Toronto': 'TOR',
        'Baltimore Orioles': 'BAL', 'Baltimore': 'BAL',

        # Central
        'Cleveland Guardians': 'CLE', 'Cleveland Indians': 'CLE', 'Cleveland': 'CLE',
        'Chicago White Sox': 'CHW', 'Chi White Sox': 'CHW',
        'Detroit Tigers': 'DET', 'Detroit': 'DET',
        'Kansas City Royals': 'KC', 'Kansas City': 'KC',
        'Minnesota Twins': 'MIN', 'Minnesota': 'MIN',

        # West
        'Houston Astros': 'HOU', 'Houston': 'HOU',
        'Seattle Mariners': 'SEA', 'Seattle': 'SEA',
        'Texas Rangers': 'TEX', 'Texas': 'TEX',
        'Oakland Athletics': 'OAK', 'Oakland': 'OAK',
        'Los Angeles Angels': 'LAA', 'LA Angels': 'LAA', 'Anaheim': 'LAA',

        # NL East
        'Atlanta Braves': 'ATL', 'Atlanta': 'ATL',
        'New York Mets': 'NYM', 'NY Mets': 'NYM', 'New York (NL)': 'NYM',
        'Philadelphia Phillies': 'PHI', 'Philadelphia': 'PHI',
        'Miami Marlins': 'MIA', 'Miami': 'MIA', 'Florida Marlins': 'MIA',
        'Washington Nationals': 'WAS', 'Washington': 'WAS',

        # NL Central
        'Chicago Cubs': 'CHC', 'Chi Cubs': 'CHC',
        'St. Louis Cardinals': 'STL', 'St. Louis': 'STL', 'St Louis': 'STL',
        'Milwaukee Brewers': 'MIL', 'Milwaukee': 'MIL',
        'Cincinnati Reds': 'CIN', 'Cincinnati': 'CIN',
        'Pittsburgh Pirates': 'PIT', 'Pittsburgh': 'PIT',

        # NL West
        'Los Angeles Dodgers': 'LAD', 'LA Dodgers': 'LAD',
        'San Diego Padres': 'SD', 'San Diego': 'SD',
        'San Francisco Giants': 'SF', 'San Francisco': 'SF',
        'Arizona Diamondbacks': 'ARI', 'Arizona': 'ARI',
        'Colorado Rockies': 'COL', 'Colorado': 'COL'
}

def get_team_abbr(name: str) -> str:
  if not isinstance(name, str):
    return "UNKNOWN"
  return TEAM_MAPPING.get(name.strip(), "UNKNOWN")
