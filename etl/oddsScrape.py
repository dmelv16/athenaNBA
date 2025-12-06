import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import time
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================
BASE_URL = "https://sportsbook-nash.draftkings.com"
NBA_LEAGUE_ID = 42648
MOUNTAIN_TZ = ZoneInfo("America/Denver")
REQUEST_DELAY = 1.0  # seconds between API calls

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:145.0) Gecko/20100101 Firefox/145.0",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://sportsbook.draftkings.com/",
    "Content-Type": "application/json",
    "Origin": "https://sportsbook.draftkings.com",
}

# Subcategories we want
WANTED_SUBCATEGORIES = {
    "16477": "points", "12488": "points_ou",
    "16479": "rebounds", "12492": "rebounds_ou",
    "16478": "assists", "12495": "assists_ou",
    "16484": "blocks", "13780": "blocks_ou",
    "16485": "steals",
    "16483": "pts_rebs_asts", "5001": "pts_rebs_asts_ou",
    "16482": "pts_rebs", "9976": "pts_rebs_ou",
    "16481": "pts_asts", "9973": "pts_asts_ou",
    "9974": "rebs_asts_ou",
    "4511": "game_lines",
}

CATEGORIES_TO_FETCH = ["487", "1215", "1216", "1217", "1293", "583"]

# =============================================================================
# DATABASE
# =============================================================================
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "nba_props"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD")
    )

def init_database():
    """Create tables if they don't exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id SERIAL PRIMARY KEY,
            dk_event_id VARCHAR(20) UNIQUE NOT NULL,
            name VARCHAR(100) NOT NULL,
            home_team VARCHAR(50),
            away_team VARCHAR(50),
            start_time_utc TIMESTAMP WITH TIME ZONE,
            start_time_mt TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS player_props (
            id SERIAL PRIMARY KEY,
            game_id INTEGER REFERENCES games(id) ON DELETE CASCADE,
            dk_market_id VARCHAR(30),
            dk_selection_id VARCHAR(100),
            player_name VARCHAR(100),
            prop_type VARCHAR(50),
            label VARCHAR(20),
            line DECIMAL(5,1),
            odds_american VARCHAR(10),
            odds_decimal DECIMAL(6,3),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(dk_selection_id)
        );
        
        CREATE TABLE IF NOT EXISTS game_lines (
            id SERIAL PRIMARY KEY,
            game_id INTEGER REFERENCES games(id) ON DELETE CASCADE,
            dk_market_id VARCHAR(30),
            dk_selection_id VARCHAR(100),
            line_type VARCHAR(30),
            team VARCHAR(50),
            label VARCHAR(30),
            line DECIMAL(5,1),
            odds_american VARCHAR(10),
            odds_decimal DECIMAL(6,3),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(dk_selection_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_player_props_game ON player_props(game_id);
        CREATE INDEX IF NOT EXISTS idx_player_props_player ON player_props(player_name);
        CREATE INDEX IF NOT EXISTS idx_player_props_type ON player_props(prop_type);
        CREATE INDEX IF NOT EXISTS idx_game_lines_game ON game_lines(game_id);
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized")

# =============================================================================
# API FUNCTIONS
# =============================================================================
def api_request(url):
    """Make API request with rate limiting."""
    time.sleep(REQUEST_DELAY)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"  API error: {e}")
    return None

def get_nba_events():
    """Get NBA events that haven't started yet."""
    url = f"{BASE_URL}/api/sportscontent/navigation/dkusnj/v2/nav/leagues/{NBA_LEAGUE_ID}"
    data = api_request(url)
    if not data:
        return []
    
    now_utc = datetime.now(timezone.utc)
    upcoming = []
    
    for event in data.get("events", []):
        start_str = event.get("startEventDate", "")
        if not start_str:
            continue
        
        # Parse UTC time
        start_utc = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        
        # Skip games that have already started
        if start_utc <= now_utc:
            print(f"  Skipping (started): {event.get('name')}")
            continue
        
        # Convert to Mountain Time
        start_mt = start_utc.astimezone(MOUNTAIN_TZ)
        
        event["start_utc"] = start_utc
        event["start_mt"] = start_mt
        upcoming.append(event)
    
    return upcoming

def get_category_data(event_id, category_id):
    """Get markets and selections for a category."""
    url = f"{BASE_URL}/api/sportscontent/dkusnj/v1/events/{event_id}/categories/{category_id}"
    return api_request(url)

# =============================================================================
# PARSING
# =============================================================================
def parse_team_names(event_name):
    """Extract home and away team from event name."""
    if " @ " in event_name:
        parts = event_name.split(" @ ")
        return parts[1].strip(), parts[0].strip()  # home, away
    return None, None

def parse_selections(data):
    """Parse markets and selections."""
    player_props = []
    game_lines = []
    
    if not data:
        return player_props, game_lines
    
    markets = {m["id"]: m for m in data.get("markets", [])}
    
    for sel in data.get("selections", []):
        market_id = sel.get("marketId")
        market = markets.get(market_id, {})
        subcat_id = market.get("subcategoryId", "")
        
        if subcat_id not in WANTED_SUBCATEGORIES:
            continue
        
        prop_type = WANTED_SUBCATEGORIES[subcat_id]
        display_odds = sel.get("displayOdds", {})
        
        # Get player/team name
        participants = sel.get("participants", [])
        entity_name = ""
        entity_type = "player"
        for p in participants:
            if p.get("type") == "Player":
                entity_name = p.get("name", "")
                break
            elif p.get("type") == "Team":
                entity_name = p.get("name", "")
                entity_type = "team"
        
        entry = {
            "dk_market_id": market_id,
            "dk_selection_id": sel.get("id"),
            "name": entity_name or market.get("name", ""),
            "prop_type": prop_type,
            "label": sel.get("label", ""),
            "line": sel.get("milestoneValue") or sel.get("points") or sel.get("line"),
            "odds_american": display_odds.get("american", ""),
            "odds_decimal": display_odds.get("decimal"),
        }
        
        if prop_type == "game_lines":
            entry["line_type"] = classify_game_line(market.get("name", ""), sel.get("label", ""))
            entry["team"] = entity_name
            game_lines.append(entry)
        else:
            entry["player_name"] = entity_name
            player_props.append(entry)
    
    return player_props, game_lines

def classify_game_line(market_name, label):
    """Classify game line type."""
    name_lower = market_name.lower()
    label_lower = label.lower()
    if "spread" in name_lower:
        return "spread"
    elif "total" in name_lower or "over" in label_lower or "under" in label_lower:
        return "total"
    elif "moneyline" in name_lower or label_lower in ["", market_name.lower()]:
        return "moneyline"
    return "other"

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================
def upsert_game(cur, event):
    """Insert or update game, return game_id."""
    home, away = parse_team_names(event.get("name", ""))
    
    cur.execute("""
        INSERT INTO games (dk_event_id, name, home_team, away_team, start_time_utc, start_time_mt, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (dk_event_id) DO UPDATE SET
            name = EXCLUDED.name,
            start_time_utc = EXCLUDED.start_time_utc,
            start_time_mt = EXCLUDED.start_time_mt,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id
    """, (
        str(event["id"]),
        event.get("name"),
        home, away,
        event["start_utc"],
        event["start_mt"].replace(tzinfo=None)
    ))
    
    return cur.fetchone()[0]

def upsert_player_props(cur, game_id, props):
    """Bulk upsert player props."""
    if not props:
        return 0
    
    values = [(
        game_id,
        p["dk_market_id"],
        p["dk_selection_id"],
        p["player_name"],
        p["prop_type"],
        p["label"],
        p["line"],
        p["odds_american"],
        p["odds_decimal"]
    ) for p in props]
    
    execute_values(cur, """
        INSERT INTO player_props (game_id, dk_market_id, dk_selection_id, player_name, prop_type, label, line, odds_american, odds_decimal)
        VALUES %s
        ON CONFLICT (dk_selection_id) DO UPDATE SET
            line = EXCLUDED.line,
            odds_american = EXCLUDED.odds_american,
            odds_decimal = EXCLUDED.odds_decimal,
            updated_at = CURRENT_TIMESTAMP
    """, values)
    
    return len(values)

def upsert_game_lines(cur, game_id, lines):
    """Bulk upsert game lines."""
    if not lines:
        return 0
    
    values = [(
        game_id,
        l["dk_market_id"],
        l["dk_selection_id"],
        l["line_type"],
        l["team"],
        l["label"],
        l["line"],
        l["odds_american"],
        l["odds_decimal"]
    ) for l in lines]
    
    execute_values(cur, """
        INSERT INTO game_lines (game_id, dk_market_id, dk_selection_id, line_type, team, label, line, odds_american, odds_decimal)
        VALUES %s
        ON CONFLICT (dk_selection_id) DO UPDATE SET
            line = EXCLUDED.line,
            odds_american = EXCLUDED.odds_american,
            odds_decimal = EXCLUDED.odds_decimal,
            updated_at = CURRENT_TIMESTAMP
    """, values)
    
    return len(values)

# =============================================================================
# MAIN
# =============================================================================
def scrape_and_upload():
    print("="*60)
    print(f"DRAFTKINGS NBA SCRAPER - {datetime.now(MOUNTAIN_TZ).strftime('%Y-%m-%d %I:%M %p MT')}")
    print("="*60)
    
    # Initialize DB
    init_database()
    
    # Get upcoming games
    print("\nFetching NBA events...")
    events = get_nba_events()
    print(f"Found {len(events)} upcoming games")
    
    if not events:
        print("No upcoming games to scrape")
        return
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    total_props = 0
    total_lines = 0
    
    for event in events:
        event_id = event["id"]
        print(f"\n{event['name']}")
        print(f"  Start: {event['start_mt'].strftime('%Y-%m-%d %I:%M %p MT')}")
        
        # Upsert game
        game_id = upsert_game(cur, event)
        
        all_props = []
        all_lines = []
        
        # Fetch each category
        for cat_id in CATEGORIES_TO_FETCH:
            data = get_category_data(event_id, cat_id)
            if data:
                props, lines = parse_selections(data)
                all_props.extend(props)
                all_lines.extend(lines)
        
        # Upsert to DB
        props_count = upsert_player_props(cur, game_id, all_props)
        lines_count = upsert_game_lines(cur, game_id, all_lines)
        
        print(f"  Saved: {props_count} player props, {lines_count} game lines")
        total_props += props_count
        total_lines += lines_count
        
        conn.commit()
    
    cur.close()
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {total_props} player props, {total_lines} game lines")
    print("="*60)

if __name__ == "__main__":
    scrape_and_upload()