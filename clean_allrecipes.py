"""
Clean the Kaggle AllRecipes dataset for the Recipe-Generator project.

Usage (from the repo root):
    python clean_allrecipes.py allrecipes.csv data/cleaned/allrecipes_cleaned.csv
"""
import re
import sys
import pandas as pd

# Fix characters that were exported as "<U+2153>" text (a common R export artifact)
UNICODE_FIXES = {
    "<U+2153>": "1/3", "<U+2154>": "2/3", "<U+215B>": "1/8",
    "<U+215C>": "3/8", "<U+215D>": "5/8", "<U+215E>": "7/8",
    "<U+2009>": " ",
}
# Real unicode fraction characters -> plain text
FRACTIONS = {"½": "1/2", "¼": "1/4", "¾": "3/4", "⅓": "1/3", "⅔": "2/3",
             "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8"}

UNITS = r"(cups?|tablespoons?|teaspoons?|tbsp|tsp|ounces?|oz|pounds?|lbs?|pints?|quarts?|gallons?|" \
        r"cloves?|slices?|pinch(?:es)?|dash(?:es)?|cans?|packages?|pieces?|sticks?|sprigs?|" \
        r"grams?|g|kg|ml|liters?|l)"
PREP_WORDS = r"\b(chopped|diced|minced|sliced|shredded|grated|crushed|melted|softened|beaten|" \
             r"peeled|cubed|halved|fresh|freshly|ground|large|small|medium|finely|roughly|thinly|" \
             r"divided|drained|rinsed|packed|cooked|uncooked|optional)\b"


def fix_text(s):
    if pd.isna(s):
        return s
    for bad, good in {**UNICODE_FIXES, **FRACTIONS}.items():
        s = s.replace(bad, good)
    return re.sub(r"\s+", " ", s).strip()


SINGULAR = {"eggs": "egg", "onions": "onion", "tomatoes": "tomato", "potatoes": "potato",
            "carrots": "carrot", "lemons": "lemon", "limes": "lime", "apples": "apple",
            "bananas": "banana", "mushrooms": "mushroom", "peppers": "pepper",
            "walnuts": "walnut", "pecans": "pecan", "almonds": "almond", "oranges": "orange"}


def normalize_ingredient(raw):
    """'2 cups chopped Onions, divided' -> ['onion']  (simple rule-based version).
    Returns a list because 'salt and pepper' is really two ingredients."""
    pieces = re.split(r"\band\b", raw.lower()) if re.search(r"\bsalt and\b", raw.lower()) else [raw.lower()]
    out = []
    for piece in pieces:
        name = _normalize_one(piece)
        if name:
            out.append(SINGULAR.get(name, name))
    return out


def _normalize_one(raw):
    s = raw.lower()
    s = re.sub(r"\(.*?\)", " ", s)            # drop parentheses: "(6 inch)", "(e.g. Tabasco)"
    s = s.split(",")[0]                        # drop trailing notes: ", or to taste"
    s = re.sub(r"[\d/.\-]+", " ", s)           # drop quantities
    s = re.sub(rf"\b{UNITS}\b", " ", s)        # drop units
    s = re.sub(PREP_WORDS, " ", s)             # drop prep words
    s = re.sub(r"[^a-z\s]", " ", s)            # drop leftover symbols (™, etc.)
    s = re.sub(r"\b(of|and|or|to|taste|for|the|a)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_minutes(text):
    """'1 hr 5 mins' / '30 mins' / '2 days' -> minutes (int) or None"""
    if not isinstance(text, str):
        return None
    total, found = 0, False
    for num, unit in re.findall(r"(\d+)\s*(day|hr|min)", text):
        found = True
        total += int(num) * {"day": 1440, "hr": 60, "min": 1}[unit]
    return total if found else None


def parse_process(p):
    """Split 'prep: 30 mins,cook: 15 mins,total: 45 mins,Servings: 4,Yield: ...' into columns."""
    out = {}
    for key in ["prep", "cook", "additional", "total"]:
        m = re.search(rf"{key}:\s*([^,]+)", p)
        out[f"{key}_minutes"] = parse_minutes(m.group(1)) if m else None
    m = re.search(r"Servings:\s*(\d+)", p)
    out["servings"] = int(m.group(1)) if m else None
    return out


def main(src, dst):
    # File is Windows-1252 encoded, not UTF-8
    df = pd.read_csv(src, encoding="cp1252")
    print(f"Loaded {len(df)} rows")

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    df = df.rename(columns={"name": "recipe_name", "ingredient": "ingredients_raw",
                            "n_rater": "num_ratings", "n_reviewer": "num_reviews",
                            "group": "category_path"})

    # Remove exact duplicate rows, then duplicate recipe names (keep the one with most reviews)
    df = df.drop_duplicates()
    df = df.sort_values("num_reviews", ascending=False).drop_duplicates("recipe_name").sort_index()
    print(f"After removing duplicates: {len(df)} rows")

    # Fix text encoding artifacts
    for col in ["recipe_name", "summary", "process", "ingredients_raw"]:
        df[col] = df[col].map(fix_text)

    # Ratings: 0 means "no ratings yet", not a real 0-star rating
    df["rating"] = df["rating"].where(df["num_ratings"] > 0)

    # Categories
    parts = df["category_path"].str.split(".")
    df["main_category"] = parts.str[0]
    df["sub_category"] = parts.str[1]

    # Ingredients: raw list + normalized list (ingredient names only)
    raw_lists = df["ingredients_raw"].str.split(r"\s\+\s", regex=True)
    df["ingredients_raw"] = raw_lists.map(lambda L: " | ".join(i.strip() for i in L))
    clean_lists = raw_lists.map(lambda L: [n for i in L for n in normalize_ingredient(i)])
    clean_lists = clean_lists.map(lambda L: sorted(set(L)))
    df["ingredients"] = clean_lists.map(lambda L: " | ".join(L))
    df["num_ingredients"] = clean_lists.map(len)

    # Time / servings columns from the "process" text
    df = pd.concat([df, df["process"].apply(lambda p: pd.Series(parse_process(p)))], axis=1)

    # Drop rows with no usable ingredients
    df = df[df["num_ingredients"] > 0]

    final_cols = ["recipe_name", "main_category", "sub_category", "rating", "num_ratings",
                  "num_reviews", "summary", "prep_minutes", "cook_minutes", "additional_minutes",
                  "total_minutes", "servings", "ingredients", "ingredients_raw", "num_ingredients"]
    df = df[final_cols].reset_index(drop=True)

    df.to_csv(dst, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows x {len(df.columns)} columns to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
