with open('docs/gospel_of_sri_ramakrishna.txt', encoding='utf-8') as f:
    text = f.read()

print(f"File size : {len(text):,} characters")
print(f"Lines     : {text.count(chr(10)):,}")

chapters = [l for l in text.splitlines() if l.startswith('VOLUME_')]
print(f"Chapters  : {len(chapters)}")

checks = [
    ("Birth date",     "February 18, 1836"),
    ("Father name",    "Kshudiram"),
    ("Childhood name", "Gadadhar"),
    ("Vivekananda",    "Narendra"),
    ("Holy Mother",    "Sarada Devi"),
    ("Last chapter",   "after_the_passing_away"),
    ("Glossary",       "GLOSSARY"),
    ("Chronology",     "CHRONOLOGY"),
    ("Vol1 Ch1",       "Master and Disciple"),
    ("Vol2 last",      "After the Passing Away"),
]
print()
for label, term in checks:
    status = "OK" if term in text else "MISSING"
    print(f"  {label:20} {status}  ({term})")
