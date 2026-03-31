import requests
import re
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from nltk.stem import PorterStemmer

# ─────────────────────────────────────────────
# Configuration and Constants
# ─────────────────────────────────────────────
BASE_URL = "https://www.themealdb.com/api/json/v1/1"
stemmer = PorterStemmer()

CUISINES = [
    "American", "British", "Canadian", "Chinese", "Croatian", "Dutch",
    "Egyptian", "Filipino", "French", "Greek", "Indian", "Irish",
    "Italian", "Jamaican", "Japanese", "Kenyan", "Malaysian", "Mexican",
    "Moroccan", "Polish", "Portuguese", "Russian", "Spanish", "Thai",
    "Tunisian", "Turkish", "Ukrainian", "Vietnamese",
]

SUGGESTION_INGREDIENTS = [
    "chicken", "eggs", "garlic", "onion", "tomato", "pasta", "rice",
    "cheese", "potato", "lemon", "butter", "beef", "mushroom",
    "spinach", "carrot", "ginger", "milk", "flour", "olive oil",
]

# ─────────────────────────────────────────────
# API Request Layer (responsible for communicating with TheMealDB)
# ─────────────────────────────────────────────

def search_by_ingredient(ingredient: str):
    """Filter recipes by a single ingredient"""
    try:
        r = requests.get(f"{BASE_URL}/filter.php", params={"i": ingredient}, timeout=8)
        data = r.json()
        return data.get("meals") or []
    except Exception:
        return []

def search_by_area(area: str):
    """Filter recipes by a single area"""
    try:
        r = requests.get(f"{BASE_URL}/filter.php", params={"a": area}, timeout=8)
        data = r.json()
        return data.get("meals") or []
    except Exception:
        return []

def get_meal_detail(meal_id: str):
    """Get the complete details of a specific recipe"""
    try:
        r = requests.get(f"{BASE_URL}/lookup.php", params={"i": meal_id}, timeout=8)
        data = r.json()
        meals = data.get("meals")
        return meals[0] if meals else None
    except Exception:
        return None

def fetch_recipes_parallel(recipe_ids):
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Fetch the details of all recipes in parallel to speed up the process
        details = list(executor.map(get_meal_detail, recipe_ids))
    return [d for d in details if d]

# ─────────────────────────────────────────────
# Data Processing Layer (responsible for processing and cleaning the data returned by the API, preparing it for Streamlit display)
# ─────────────────────────────────────────────

def get_stemmed_tokens(text: str) -> set:
    if not text: 
        return set()

    text = re.sub(r'\(.*\)', '', text.lower())
    text = re.sub(r'[^a-z\s]', '', text)

    stop_words = {'chopped', 'sliced', 'diced', 'fresh', 'dried', 'large', 'small', 'minced',
                  'powdered', 'ground', 'grated', 'shredded', 'crushed'}

    tokens = text.split()
    stemmed = {stemmer.stem(word) for word in tokens if word not in stop_words}
    return stemmed


def is_match(item_name: str, fridge_item: str) -> bool:
    """
    Determine if a recipe ingredient matches an item in the fridge using fuzzy matching.
    """
    # Get the stemmed tokens for both the recipe ingredient and the fridge item
    recipe_stems = get_stemmed_tokens(item_name)
    fridge_stems = get_stemmed_tokens(fridge_item)
    
    if not recipe_stems or not fridge_stems:
        return False
        
    # check if the stemmed tokens of the fridge item are a subset of the recipe ingredient's tokens, or vice versa.
    # for example, "chopped garlic" would have the stem "garlic", which would match with "garlic" in the fridge.
    return fridge_stems.issubset(recipe_stems) or recipe_stems.issubset(fridge_stems)


def extract_ingredients(meal: dict) -> list[dict]:
    """Clean the scattered ingredient fields returned by the API into a standardized list format"""
    items = []
    for i in range(1, 21):
        name = (meal.get(f"strIngredient{i}") or "").strip()
        measure = (meal.get(f"strMeasure{i}") or "").strip()
        if name:
            items.append({"name": name.lower(), "measure": measure})
    return items


def calc_match(meal_detail: dict, fridge: list[str]):
    """
    Compare the required ingredients of a recipe with the available ingredients in the fridge, 
    and return the match percentage, list of available ingredients, and list of missing ingredients.
    """
    recipe_ings = extract_ingredients(meal_detail)
    if not recipe_ings:
        return 0, [], []
    
    fridge_lower = [f.lower() for f in fridge]
    have, missing = [], []
    
    for item in recipe_ings:
        matched = any(
            item["name"] in f or f in item["name"]
            for f in fridge_lower
        )
        if matched:
            have.append(item)
        else:
            missing.append(item)
            
    pct = round(len(have) / len(recipe_ings) * 100) if recipe_ings else 0
    return pct, have, missing


def get_ingredient_frequency(recipes: list[dict]):
    """Count the frequency of the appearance of ingredients in all matching recipes for data visualization"""
    ing_freq = defaultdict(int)
    for r in recipes:
        all_ings = r.get("have", []) + r.get("missing", [])
        
        for item in all_ings:
            ing_freq[item["name"]] += 1
    
    df = pd.DataFrame(ing_freq.items(), columns=["Ingredient", "Appears in # recipes"])
    return df.sort_values("Appears in # recipes", ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────
# Others
# ─────────────────────────────────────────────

def pct_color(pct: int) -> str:
    """Return the corresponding status icon based on the match percentage"""
    if pct >= 80:
        return "🟢"
    if pct >= 60:
        return "🟡"
    if pct >= 40:
        return "🟠"
    return "🔴"