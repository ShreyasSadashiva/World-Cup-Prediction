"""
Maps football-data.org team names → canonical names used in the martj42 CSV dataset.
Add entries as needed when the seeder reports unmatched names.
"""

# football-data.org name  →  martj42 CSV name
API_TO_CSV: dict[str, str] = {
    "United States":           "United States",
    "USA":                     "United States",
    "Korea Republic":          "South Korea",
    "Republic of Korea":       "South Korea",
    "IR Iran":                 "Iran",
    "Türkiye":                 "Turkey",
    "Côte d'Ivoire":           "Ivory Coast",
    "DR Congo":                "DR Congo",
    "Bosnia and Herzegovina":  "Bosnia-Herzegovina",
    "Bosnia & Herzegovina":    "Bosnia-Herzegovina",
    "Cape Verde":              "Cape Verde Islands",
    "North Macedonia":         "North Macedonia",
    "Trinidad and Tobago":     "Trinidad and Tobago",
    "Czech Republic":          "Czech Republic",
    "Czechia":                 "Czech Republic",
    "Cape Verde":              "Cape Verde",
    "Congo DR":                "DR Congo",
    "Dominican Republic":      "Dominican Republic",
    "Guinea-Bissau":           "Guinea-Bissau",
    "São Tomé and Príncipe":   "Sao Tome and Principe",
    "Antigua and Barbuda":     "Antigua and Barbuda",
    "Saint Kitts and Nevis":   "Saint Kitts and Nevis",
    "Saint Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Solomon Islands":         "Solomon Islands",
    "Papua New Guinea":        "Papua New Guinea",
    "New Zealand":             "New Zealand",
    "China PR":                "China",
    "Chinese Taipei":          "Taiwan",
    "Korea DPR":               "North Korea",
    "Hong Kong":               "Hong Kong",
    "Macao":                   "Macau",
}


def normalize(api_name: str) -> str:
    """Return canonical CSV name, falling back to the API name unchanged."""
    return API_TO_CSV.get(api_name, api_name)
