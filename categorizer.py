#!/usr/bin/env python3
"""
Video Categorizer v2 — Joy Of Gaming
Clasificación manual precisa basada en los títulos reales del canal.
"""

import re
import pandas as pd
from tqdm import tqdm

INPUT_CSV = "videos_raw_data.csv"
OUTPUT_CSV = "videos_categorized.csv"

# ---------------------------------------------------------------------------
# Diccionario de juegos con género, orden = prioridad de matching
# Los patrones más específicos van primero para evitar falsos positivos
# ---------------------------------------------------------------------------
GAME_PATTERNS = [
    # === SPORTS / FIFA ===
    (r"FIFA 22", "FIFA 22", "sports"),
    (r"FIFA 23", "FIFA 23", "sports"),
    (r"FIFA 21", "FIFA 21", "sports"),
    (r"EA FC 25", "EA FC 25", "sports"),
    (r"EA FC 24", "EA FC 24", "sports"),
    (r"EA FC", "EA FC", "sports"),
    (r"eFootball|PES 202[0-9]|PES 2021|PES 2022", "PES / eFootball", "sports"),
    (r"FIFA STREET|BRAZILIAN FAVELA FIFA", "FIFA Street", "sports"),
    (r"NBA 2K2[0-9]|NBA 2K21|NBA 2K22|NBA 2K23|NBA 2K24", "NBA 2K", "sports"),
    (r"UFC [0-9]|UFC\b|BACKYARD FIGHTING", "UFC", "sports"),
    (r"WWE 2K(?:19|22|23|Battlegrounds| )", "WWE 2K", "sports"),
    (r"Madden NFL|MADDEN", "Madden NFL", "sports"),
    (r"MLB The Show", "MLB The Show", "sports"),
    (r"OLYMPIC GAMES", "Olympic Games", "sports"),

    # === RACING ===
    (r"Gran Turismo 7|GT7", "Gran Turismo 7", "racing"),
    (r"Gran Turismo Sport|GT Sport|GRAN TURISMO SPORT", "Gran Turismo Sport", "racing"),
    (r"DRIVECLUB|DriveClub|Driveclub", "DriveClub", "racing"),
    (r"RIDE 5|Ride 5|RIDE5", "RIDE 5", "racing"),
    (r"RIDE 4|Ride 4|RIDE4", "RIDE 4", "racing"),
    (r"DIRT 5|Dirt 5|DiRT 5", "DIRT 5", "racing"),
    (r"DIRT Rally 2\.0|DiRT Rally 2\.0|DIRT Rally 2", "DIRT Rally 2.0", "racing"),
    (r"DIRT Rally|DiRT Rally", "DIRT Rally", "racing"),
    (r"Need for Speed Unbound|NFS Unbound", "Need for Speed Unbound", "racing"),
    (r"Need for Speed Heat|NFS Heat", "Need for Speed Heat", "racing"),
    (r"Need for Speed Payback|NFS Payback", "Need for Speed Payback", "racing"),
    (r"Need for Speed Rivals|NFS Rivals|NEED FOR SPEED RIVALS", "Need for Speed Rivals", "racing"),
    (r"Need for Speed Hot Pursuit", "Need for Speed Hot Pursuit", "racing"),
    (r"Need for Speed|NFS", "Need for Speed", "racing"),
    (r"Forza Horizon 5", "Forza Horizon 5", "racing"),
    (r"Forza Horizon 4", "Forza Horizon 4", "racing"),
    (r"Forza Motorsport|FORZA", "Forza Motorsport", "racing"),
    (r"F1 25", "F1 25", "racing"),
    (r"F1 2[0-9]|F1 23|FORMULA 1", "F1 (Series)", "racing"),
    (r"WRC [0-9]|World Rally Championship|WRC\b", "WRC", "racing"),
    (r"MotoGP|MOTOGP", "MotoGP", "racing"),
    (r"MXGP|Motocross", "MXGP / Motocross", "racing"),
    (r"Monster Energy Supercross|Supercross", "Monster Energy Supercross", "racing"),
    (r"NASCAR|Nascar", "NASCAR Heat", "racing"),
    (r"Assetto Corsa", "Assetto Corsa", "racing"),
    (r"Hot Wheels Unleashed|HOT WHEELS|Hot Wheels", "Hot Wheels Unleashed", "racing"),
    (r"Monster Jam|MONSTER JAM", "Monster Jam", "racing"),
    (r"The Crew MotorFest|CREW MOTORFEST", "The Crew Motorfest", "racing"),
    (r"The Crew 2", "The Crew 2", "racing"),
    (r"Wreckfest|WRECKFEST", "Wreckfest", "racing"),
    (r"Burnout Paradise", "Burnout Paradise", "racing"),
    (r"Project Cars|PROJECT CARS", "Project Cars", "racing"),
    (r"Test Drive Unlimited", "Test Drive Unlimited", "racing"),
    (r"DAKAR Desert Rally|DAKAR", "DAKAR Desert Rally", "racing"),
    (r"NO FORZA buggy", "Forza Motorsport", "racing"),

    # === EXTREME SPORTS ===
    (r"Riders Republic|RIDERS REPUBLIC", "Riders Republic", "extreme_sports"),
    (r"Steep\b|STEEP", "Steep", "extreme_sports"),
    (r"Tony Hawk|TONY HAWK", "Tony Hawk's Pro Skater", "extreme_sports"),
    (r"Skate\b|SKATE\b", "Skate", "extreme_sports"),
    (r"SURFING GAME", "Surfing Game", "extreme_sports"),
    (r"SNOW SPORTS GAME|EXTREME SPORTS GAME|CRAZIEST SPORTS GAME", "Extreme Sports", "extreme_sports"),

    # === SHOOTER ===
    (r"Call of Duty.*Modern Warfare 3|Modern Warfare 3|MW3", "Call of Duty: MW3", "shooter"),
    (r"Call of Duty.*Modern Warfare 2|Modern Warfare 2|MW2\b", "Call of Duty: MW2", "shooter"),
    (r"Call of Duty.*Modern Warfare|Modern Warfare|Clean House", "Call of Duty: Modern Warfare", "shooter"),
    (r"Call of Duty.*Black Ops 6|BLACK OPS 6|BO6", "Call of Duty: Black Ops 6", "shooter"),
    (r"Call of Duty.*Black Ops Cold War|BLACK OPS COLD WAR|BOCW", "Call of Duty: Black Ops Cold War", "shooter"),
    (r"Call of Duty.*Vanguard", "Call of Duty: Vanguard", "shooter"),
    (r"Call of Duty.*WWII|Call of Duty.*WW2", "Call of Duty: WWII", "shooter"),
    (r"Call of Duty.*Ghosts", "Call of Duty: Ghosts", "shooter"),
    (r"Call of Duty|COD\b", "Call of Duty", "shooter"),
    (r"BOMBING SADDAM|SADDAM HUSSEIN", "Call of Duty: Black Ops 6", "shooter"),
    (r"US Embassy Siege|LONDON ATTACK|UKRAINIAN HUNT|Ghillied up", "Call of Duty: Modern Warfare", "shooter"),
    (r"Battlefield 2042", "Battlefield 2042", "shooter"),
    (r"Battlefield V|BATTLEFIELD V|Battlefield 5", "Battlefield V", "shooter"),
    (r"BATTLEFIELD 4|Battlefield 4", "Battlefield 4", "shooter"),
    (r"Battlefield 1|BATTLEFIELD 1", "Battlefield 1", "shooter"),
    (r"Battlefield", "Battlefield", "shooter"),
    (r"Far Cry 6|FAR CRY 6", "Far Cry 6", "shooter"),
    (r"Far Cry 5|FAR CRY 5", "Far Cry 5", "shooter"),
    (r"Far Cry Primal|FAR CRY PRIMAL", "Far Cry Primal", "shooter"),
    (r"Far Cry [0-9]|Far Cry|FAR CRY", "Far Cry", "shooter"),
    (r"METRO EXODUS|Metro Exodus", "Metro Exodus", "shooter"),
    (r"Sniper Elite 5|SNIPER ELITE 5", "Sniper Elite 5", "shooter"),
    (r"Sniper Elite 4|SNIPER ELITE 4", "Sniper Elite 4", "shooter"),
    (r"Sniper Ghost Warrior", "Sniper Ghost Warrior", "shooter"),
    (r"World War Z", "World War Z", "shooter"),
    (r"Zombie Army 4|Zombie Army", "Zombie Army 4", "shooter"),
    (r"Back 4 Blood", "Back 4 Blood", "shooter"),
    (r"Enlisted\b", "Enlisted", "shooter"),
    (r"Halo Infinite", "Halo Infinite", "shooter"),
    (r"Doom Eternal|DOOM", "Doom", "shooter"),
    (r"Titanfall 2", "Titanfall 2", "shooter"),
    (r"Robocop Rogue City|ROBOCOP", "Robocop: Rogue City", "shooter"),
    (r"Killzone Shadow Fall", "Killzone Shadow Fall", "shooter"),
    (r"Wolfenstein", "Wolfenstein", "shooter"),
    (r"Borderlands 3", "Borderlands 3", "shooter"),
    (r"Destiny 2", "Destiny 2", "shooter"),
    (r"Rainbow Six|R6", "Rainbow Six Siege", "shooter"),
    (r"Insurgency", "Insurgency Sandstorm", "shooter"),
    (r"Hell Let Loose", "Hell Let Loose", "shooter"),
    (r"Crysis.*Remastered|Crysis 3|CRYSIS", "Crysis Remastered", "shooter"),
    (r"PROTOTYPE|Prototype", "Prototype", "shooter"),
    (r"Payday 3|PAYDAY", "Payday 3", "shooter"),
    (r"Tom Clancy.*Division|Division 2", "The Division 2", "shooter"),
    (r"Chivalry 2|CHIVALRY", "Chivalry 2", "shooter"),
    (r"Pacific War|Vietnam War|BATTLE OF MIDWAY", "War Missions (COD)", "shooter"),
    (r"Cartel Protection|Train Chase", "Call of Duty: Modern Warfare", "shooter"),
    (r"BRUTAL SNIPING", "Sniper Elite 5", "shooter"),
    (r"FORSPOKEN|Forspoken", "Forspoken", "shooter"),

    # === ACTION / ADVENTURE ===
    (r"God of War Ragnarok|GOD OF WAR RAGNAROK", "God of War Ragnarok", "action_adventure"),
    (r"God of War|GOD OF WAR", "God of War", "action_adventure"),
    (r"Ghost of Tsushima|GHOST OF TSUSHIMA", "Ghost of Tsushima", "action_adventure"),
    (r"Spider.?Man.*Miles Morales|Miles Morales", "Spider-Man: Miles Morales", "action_adventure"),
    (r"Spider.?Man 2|SPIDER.?MAN 2", "Spider-Man 2", "action_adventure"),
    (r"Spider.?Man|SPIDER.?MAN", "Spider-Man", "action_adventure"),
    (r"Last of Us Part II|Last of Us 2|LAST OF US 2", "The Last of Us Part II", "action_adventure"),
    (r"Last of Us|LAST OF US", "The Last of Us", "action_adventure"),
    (r"Uncharted.*Lost Legacy", "Uncharted: The Lost Legacy", "action_adventure"),
    (r"Uncharted 4|UNCHARTED 4", "Uncharted 4", "action_adventure"),
    (r"Uncharted 3|UNCHARTED 3", "Uncharted 3", "action_adventure"),
    (r"Uncharted", "Uncharted", "action_adventure"),
    (r"Horizon Forbidden West", "Horizon Forbidden West", "action_adventure"),
    (r"Horizon Zero Dawn", "Horizon Zero Dawn", "action_adventure"),
    (r"Red Dead Redemption 2|RDR2|RED DEAD", "Red Dead Redemption 2", "action_adventure"),
    (r"Grand Theft Auto.*San Andreas|GTA.*San Andreas", "GTA: San Andreas", "action_adventure"),
    (r"Grand Theft Auto V|GTA V|GTA 5", "GTA V", "action_adventure"),
    (r"Grand Theft Auto|GTA\b", "Grand Theft Auto", "action_adventure"),
    (r"Assassin.?s Creed Shadows", "Assassin's Creed Shadows", "action_adventure"),
    (r"Assassin.?s Creed Mirage", "Assassin's Creed Mirage", "action_adventure"),
    (r"Assassin.?s Creed Valhalla", "Assassin's Creed Valhalla", "action_adventure"),
    (r"Assassin.?s Creed Odyssey", "Assassin's Creed Odyssey", "action_adventure"),
    (r"Assassin.?s Creed Origins", "Assassin's Creed Origins", "action_adventure"),
    (r"Assassin.?s Creed Unity", "Assassin's Creed Unity", "action_adventure"),
    (r"Assassin.?s Creed", "Assassin's Creed", "action_adventure"),
    (r"Batman Arkham Knight|BATMAN ARKHAM KNIGHT", "Batman: Arkham Knight", "action_adventure"),
    (r"BATMAN|Batman", "Batman: Arkham", "action_adventure"),
    (r"Shadow of the Tomb Raider", "Shadow of the Tomb Raider", "action_adventure"),
    (r"Rise of the Tomb Raider", "Rise of the Tomb Raider", "action_adventure"),
    (r"Tomb Raider", "Tomb Raider", "action_adventure"),
    (r"Ghost Recon Breakpoint|GHOST RECON BREAKPOINT", "Ghost Recon Breakpoint", "action_adventure"),
    (r"Ghost Recon Wildlands", "Ghost Recon Wildlands", "action_adventure"),
    (r"Ghost Recon", "Ghost Recon", "action_adventure"),
    (r"Metal Gear Solid V|METAL GEAR SOLID V", "Metal Gear Solid V", "action_adventure"),
    (r"Metal Gear Solid|METAL GEAR", "Metal Gear Solid", "action_adventure"),
    (r"Death Stranding", "Death Stranding", "action_adventure"),
    (r"Days Gone|DAYS GONE", "Days Gone", "action_adventure"),
    (r"Mad Max|MAD MAX", "Mad Max", "action_adventure"),
    (r"inFAMOUS|Infamous|INFAMOUS", "inFAMOUS Second Son", "action_adventure"),
    (r"Mafia.*Definitive|MAFIA", "Mafia: Definitive Edition", "action_adventure"),
    (r"Watch Dogs|WATCH DOGS", "Watch Dogs", "action_adventure"),
    (r"Sleeping Dogs", "Sleeping Dogs", "action_adventure"),
    (r"Just Cause 4|JUST CAUSE", "Just Cause 4", "action_adventure"),
    (r"Hitman 3|HITMAN 3", "Hitman 3", "action_adventure"),
    (r"Hitman\b|HITMAN\b", "Hitman", "action_adventure"),
    (r"Predator Hunting Grounds|Predator.*Ghost Recon|Predator", "Predator: Hunting Grounds", "action_adventure"),
    (r"The Order 1886|ORDER 1886", "The Order: 1886", "action_adventure"),
    (r"Kena Bridge of Spirits|KENA", "Kena: Bridge of Spirits", "action_adventure"),
    (r"Returnal\b|RETURNAL", "Returnal", "action_adventure"),
    (r"Sifu\b", "Sifu", "action_adventure"),
    (r"Stray\b|STRAY\b", "Stray", "action_adventure"),
    (r"A Plague Tale|PLAGUE TALE", "A Plague Tale", "action_adventure"),
    (r"Gotham Knights", "Gotham Knights", "action_adventure"),
    (r"Suicide Squad", "Suicide Squad", "action_adventure"),
    (r"It Takes Two", "It Takes Two", "action_adventure"),
    (r"Control\b", "Control", "action_adventure"),
    (r"Dying Light 2", "Dying Light 2", "action_adventure"),
    (r"Dying Light", "Dying Light", "action_adventure"),
    (r"Saints Row", "Saints Row", "action_adventure"),
    (r"Ratchet.*Clank|RATCHET", "Ratchet & Clank", "action_adventure"),
    (r"Sackboy", "Sackboy", "action_adventure"),
    (r"Guardians of the Galaxy|GUARDIANS", "Guardians of the Galaxy", "action_adventure"),
    (r"Marvel.?s Avengers|AVENGERS|BLACK WIDOW.*TASKMASTER", "Marvel's Avengers", "action_adventure"),
    (r"Star Wars Outlaws", "Star Wars Outlaws", "action_adventure"),
    (r"Star Wars Battlefront II|Star Wars Battlefront", "Star Wars Battlefront II", "action_adventure"),
    (r"Star Wars Squadrons|STAR WARS Squadrons", "Star Wars Squadrons", "action_adventure"),
    (r"Shadow of the Colossus|Shadow Of The Colossus|SHADOW OF THE COLOSSUS", "Shadow of the Colossus", "action_adventure"),
    (r"Detroit Become Human|DETROIT", "Detroit: Become Human", "action_adventure"),
    (r"Deliver Us The Moon|DELIVER US", "Deliver Us The Moon", "action_adventure"),
    (r"Black Myth.?Wukong", "Black Myth: Wukong", "action_adventure"),
    (r"Ace Combat 7|ACE COMBAT", "Ace Combat 7", "action_adventure"),
    (r"Lost Judgment|LOST JUDGMENT", "Lost Judgment", "action_adventure"),
    (r"Vampyr\b", "Vampyr", "action_adventure"),
    (r"DC Universe|BLACK ADAM|JUSTICE LEAGUE", "DC Universe Online", "action_adventure"),
    (r"Worms Rumble", "Worms Rumble", "action_adventure"),
    (r"SCARLET NEXUS", "Scarlet Nexus", "action_adventure"),
    (r"Oddworld Soulstorm|ODDWORLD", "Oddworld: Soulstorm", "action_adventure"),
    (r"AMONG US", "Among Us", "action_adventure"),
    (r"One Piece Odyssey|ONE PIECE", "One Piece Odyssey", "action_adventure"),
    (r"Monster Hunter", "Monster Hunter", "action_adventure"),

    # === RPG ===
    (r"Witcher 3|WITCHER 3", "The Witcher 3", "rpg"),
    (r"Cyberpunk 2077|CYBERPUNK", "Cyberpunk 2077", "rpg"),
    (r"Elden Ring|ELDEN RING", "Elden Ring", "rpg"),
    (r"Dark Souls", "Dark Souls", "rpg"),
    (r"Bloodborne", "Bloodborne", "rpg"),
    (r"Demon.?s? Souls|DEMON.?S? SOULS", "Demon's Souls", "rpg"),
    (r"Final Fantasy", "Final Fantasy", "rpg"),
    (r"Hogwarts Legacy|HOGWARTS", "Hogwarts Legacy", "rpg"),
    (r"Dragon.?s Dogma", "Dragon's Dogma 2", "rpg"),
    (r"Diablo [IV4]|DIABLO", "Diablo IV", "rpg"),
    (r"Star Wars Jedi.*Survivor", "Star Wars Jedi: Survivor", "rpg"),
    (r"Star Wars Jedi.*Fallen|STAR WARS Jedi Fallen", "Star Wars Jedi: Fallen Order", "rpg"),
    (r"Baldur.?s Gate 3", "Baldur's Gate 3", "rpg"),
    (r"Kingdom Come Deliverance", "Kingdom Come Deliverance", "rpg"),
    (r"Skyrim", "Skyrim", "rpg"),
    (r"Mass Effect", "Mass Effect", "rpg"),

    # === HORROR ===
    (r"Resident Evil Village|RE Village", "Resident Evil Village", "horror"),
    (r"Resident Evil 4\b|RE4\b", "Resident Evil 4", "horror"),
    (r"Resident Evil 3|RE3|RESIDENT EVIL 3", "Resident Evil 3", "horror"),
    (r"Resident Evil 2|RE2|RESIDENT EVIL 2", "Resident Evil 2", "horror"),
    (r"Resident Evil Requiem|RESIDENT EVIL REQUIEM", "Resident Evil Requiem", "horror"),
    (r"Resident Evil|RESIDENT EVIL", "Resident Evil", "horror"),
    (r"The Evil Within|EVIL WITHIN", "The Evil Within", "horror"),
    (r"Dead Space.*Remake|DEAD SPACE", "Dead Space Remake", "horror"),
    (r"Blair Witch|BLAIR WITCH", "Blair Witch", "horror"),
    (r"Until Dawn", "Until Dawn", "horror"),
    (r"Silent Hill", "Silent Hill", "horror"),
    (r"Alien Isolation", "Alien Isolation", "horror"),
    (r"Outlast|OUTLAST", "Outlast", "horror"),
    (r"Little Nightmares", "Little Nightmares", "horror"),
    (r"Callisto Protocol", "The Callisto Protocol", "horror"),
    (r"Observer.*Redux|OBSERVER|HORROR GAME.*NIGHTMARES", "Observer: System Redux", "horror"),

    # === FIGHTING ===
    (r"Mortal Kombat 1[^1]|MORTAL KOMBAT 1[^1]", "Mortal Kombat 1", "fighting"),
    (r"Mortal Kombat 11|MORTAL KOMBAT 11|Mortal Kombat X", "Mortal Kombat 11", "fighting"),
    (r"Mortal Kombat|MORTAL KOMBAT", "Mortal Kombat", "fighting"),
    (r"Street Fighter 6", "Street Fighter 6", "fighting"),
    (r"Tekken 8", "Tekken 8", "fighting"),
    (r"Tekken 7|TEKKEN", "Tekken 7", "fighting"),
    (r"Dragon Ball", "Dragon Ball", "fighting"),
    (r"VIKINGS VS SAMURAI", "For Honor", "fighting"),

    # === PLATFORMER ===
    (r"Crash Bandicoot|CRASH BANDICOOT", "Crash Bandicoot", "platformer"),
    (r"Astro Bot|ASTRO BOT", "Astro Bot", "platformer"),
    (r"Spyro", "Spyro", "platformer"),

    # === SURVIVAL ===
    (r"The Forest|THE FOREST", "The Forest", "survival"),
    (r"Subnautica", "Subnautica", "survival"),
    (r"The Day Before|DAY BEFORE", "The Day Before", "survival"),
    (r"DayZ", "DayZ", "survival"),

    # === SANDBOX ===
    (r"Minecraft", "Minecraft", "sandbox"),
    (r"No Man.?s Sky", "No Man's Sky", "sandbox"),
    (r"Dreams\b", "Dreams", "sandbox"),
    (r"Teardown", "Teardown", "sandbox"),

    # === MISC known titles ===
    (r"Hellblade 2|HELLBLADE", "Hellblade 2", "action_adventure"),
    (r"Alan Wake 2", "Alan Wake 2", "horror"),
    (r"Alan Wake", "Alan Wake", "horror"),
    (r"The Matrix", "The Matrix Awakens", "other"),
    (r"PRAGMATA|Pragmata", "Pragmata", "other"),
    (r"AVATAR GAME", "Avatar", "action_adventure"),
    (r"FORTNITE|Fortnite|ZERO BUILDING", "Fortnite", "battle_royale"),
    (r"Apex Legends", "Apex Legends", "battle_royale"),
    (r"Warzone\b|WARZONE", "Call of Duty: Warzone", "battle_royale"),
    (r"PUBG", "PUBG", "battle_royale"),
    (r"BULLY|Opcion bully", "Bully", "action_adventure"),
    (r"Robocop|ROBOCOP", "Robocop: Rogue City", "shooter"),
]


# ---------------------------------------------------------------------------
# Formato de video — nombres claros en español
# ---------------------------------------------------------------------------
FORMAT_LABELS = {
    "gameplay": "Gameplay Puro",
    "story_mode": "Modo Historia",
    "showcase": "Showcase Visual",
    "full_match": "Partido Completo",
    "compilation": "Compilación",
    "stealth": "Stealth / Sigilo",
    "free_roam": "Free Roam / Exploración",
    "cinematic": "Cinemático / Trailer",
    "competitive": "Competitivo / Online",
    "review": "Review / Opinión",
    "challenge": "Challenge / Reto",
    "first_person": "Primera Persona POV",
    "other": "Otro",
}

# ---------------------------------------------------------------------------
# Estilo visual
# ---------------------------------------------------------------------------
STYLE_LABELS = {
    "realistic": "Ultra Realista",
    "cinematic": "Cinemático",
    "competitive": "Competitivo",
    "gameplay_puro": "Gameplay Estándar",
    "tutorial": "Tutorial / Guía",
}


def classify_game(title, tags):
    """Clasifica el juego basándose en regex patterns contra título y tags."""
    if not isinstance(title, str):
        return "Unknown", "other"

    # Ignorar templates
    if "Story/Clickbait Hook" in title:
        return "Unknown (Template)", "other"

    # GoPro streams
    if "GoPro" in title or "Transmisión en directo" in title:
        return "Stream / GoPro", "other"

    # Test / internal videos
    if title.startswith("test ") or "SE PARO" in title or "HandBreakCheck" in title or title.startswith("Opcion "):
        return "Test / Interno", "other"

    # 5 robos
    if "robos hechos" in title.lower():
        return "Otro Contenido", "other"

    combined = f"{title} {tags}" if isinstance(tags, str) else title

    for pattern, game_name, genre in GAME_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return game_name, genre

    return "Otro / No Identificado", "other"


def classify_format(title, duration_seconds):
    """Clasifica el formato del video con nombres claros."""
    t = str(title).lower()

    if any(kw in t for kw in ["match prediction", "match highlights", "full match",
                               "match simulation", "prediction highlights",
                               "copa libertadores", "champions league",
                               "premier league", "la liga", "serie a",
                               "bundesliga", "liga mx", "ligue 1",
                               "europa league", "eredivisie", "super lig",
                               "brasileirao", "liga nos", "concacaf"]):
        return "full_match"

    if any(kw in t for kw in ["trailer", "reveal", "launch trailer", "official trailer"]):
        return "cinematic"

    if any(kw in t for kw in ["all cutscenes", "full movie", "cutscene"]):
        return "cinematic"

    if any(kw in t for kw in ["review", "worth it", "honest opinion"]):
        return "review"

    if any(kw in t for kw in ["challenge", "reto", "can i ", "impossible"]):
        return "challenge"

    if any(kw in t for kw in ["stealth", "sigilo", "solo stealth"]):
        return "stealth"

    if any(kw in t for kw in ["free roam", "free roaming", "exploring", "exploration"]):
        return "free_roam"

    if any(kw in t for kw in ["first person", "pov view", "pov ", "first person pov"]):
        return "first_person"

    if any(kw in t for kw in ["domination", "multiplayer", "online", "ranked",
                               "battle royale", "rumble"]):
        return "competitive"

    if any(kw in t for kw in ["story", "campaign", "mission", "walkthrough",
                               "prison break", "prison escape", "train robbery",
                               "embassy", "chase", "escape"]):
        return "story_mode"

    if any(kw in t for kw in ["cinematic", "movie", "scene", "fight scene"]):
        return "showcase"

    # Default: gameplay puro (showcase visual de gráficos = el core del canal)
    if any(kw in t for kw in ["ultra realistic", "realistic graphics", "ultra high",
                               "incredible", "amazing", "beautiful", "stunning",
                               "best looking", "best graphics", "next gen",
                               "ray tracing", "real life", "unbelievable",
                               "blows my mind", "insane", "unreal"]):
        return "showcase"

    return "gameplay"


def classify_style(title):
    """Clasifica el estilo visual."""
    t = str(title).lower()

    if any(kw in t for kw in ["cinematic", "movie", "cutscene", "trailer", "scene"]):
        return "cinematic"
    if any(kw in t for kw in ["competitive", "ranked", "domination", "multiplayer", "online"]):
        return "competitive"
    if any(kw in t for kw in ["tutorial", "guide", "tips", "how to"]):
        return "tutorial"
    if any(kw in t for kw in ["ultra realistic", "realistic", "photorealistic",
                               "real life", "next gen", "ray tracing",
                               "incredible", "amazing", "beautiful", "stunning",
                               "best looking", "best graphics", "unbelievable",
                               "blows my mind", "insane", "unreal"]):
        return "realistic"
    return "gameplay_puro"


def main():
    print("=" * 60)
    print("  Video Categorizer v2")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    print(f"Videos cargados: {len(df)}")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Categorizando"):
        title = row.get("title", "")
        tags = row.get("tags", "")
        duration = row.get("duration_seconds", 0)

        game_name, game_genre = classify_game(title, tags)
        video_format = classify_format(title, duration)
        visual_style = classify_style(title)

        results.append({
            "game_name": game_name,
            "game_genre": game_genre,
            "video_format": video_format,
            "video_format_label": FORMAT_LABELS.get(video_format, video_format),
            "visual_style": visual_style,
            "visual_style_label": STYLE_LABELS.get(visual_style, visual_style),
        })

    cats = pd.DataFrame(results)
    df["game_name"] = cats["game_name"]
    df["game_genre"] = cats["game_genre"]
    df["video_format"] = cats["video_format"]
    df["video_format_label"] = cats["video_format_label"]
    df["visual_style"] = cats["visual_style"]
    df["visual_style_label"] = cats["visual_style_label"]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nArchivo guardado: {OUTPUT_CSV}")

    print("\n--- Distribución por Género ---")
    print(df["game_genre"].value_counts().to_string())
    print("\n--- Distribución por Formato ---")
    print(df["video_format_label"].value_counts().to_string())
    print("\n--- Distribución por Estilo Visual ---")
    print(df["visual_style_label"].value_counts().to_string())
    print(f"\n--- Top 25 Juegos ---")
    print(df["game_name"].value_counts().head(25).to_string())

    # Verificar sin clasificar
    unknown = df[df["game_name"].str.contains("Otro|Unknown|Test|Stream", na=False)]
    print(f"\n--- Sin clasificar: {len(unknown)} videos ---")
    if len(unknown) > 0:
        for _, r in unknown.head(20).iterrows():
            print(f"  [{r['game_genre']:20s}] {r['title'][:80]}")

    print("=" * 60)


if __name__ == "__main__":
    main()
