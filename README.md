# Tool calling s LLM: hledání produktu, kontrola obrázku a převod měny

Python aplikace, která komunikuje s LLM (OpenAI) a nechá model samostatně rozhodnout, jestli k odpovědi potřebuje nástroj. Aplikace nástroj skutečně spustí, výsledek vrátí modelu a ten z něj složí finální odpověď.

Typický dotaz, na který je aplikace postavená:

> Najdi nejlevnější černý Apple MacBook M5 v Německu, ověř podle obrázku, že je černý, a cenu převeď do CZK.

Model si sám rozdělí práci: vyhledá nabídky, otevře stránku produktu, zkontroluje obrázek a nakonec převede cenu.

## Obsah repozitáře

| Soubor | Účel |
|:--|:--|
| `main.py` | Hlavní logika: smyčka s LLM a spouštění nástrojů |
| `tools.py` | Implementace nástrojů, jejich validace a schémata pro LLM (oddělené od hlavní logiky) |
| `tests/` | Automatické testy (`pytest`), běží bez internetu a bez klíčů |
| `jev_filter.py` | Volitelný filtr: model Jev seřadí výsledky hledání podle požadavků (viz níže) |
| `agent_logging.py` | Nastavení logování: stručný výpis do konzole a podrobný soubor `agent.log` |
| `test_queries.py` | 5 ukázkových dotazů proti skutečnému LLM, s kontrolou, které nástroje se použily |
| `try_tool.py` | Pomocný skript: zavolá jeden nástroj ručně, bez LLM |
| `check_connection.py` | Pomocný skript: ověří, že klíč `OPENAI_API_KEY` funguje |
| `pyproject.toml` | Seznam závislostí |
| `.env.example` | Šablona pro API klíče (bez skutečných klíčů) |

## Spuštění

Potřebuješ Python 3.12+ a [uv](https://docs.astral.sh/uv/).

1. Přejdi do složky projektu:
   ```bash
   cd 01_llm_api/X_Homework/01_Homework
   ```
2. Vytvoř soubor `.env` ze šablony a vyplň své klíče:
   ```bash
   cp .env.example .env
   ```
   | Proměnná | Odkud ji vzít | Povinná |
   |:--|:--|:--|
   | `OPENAI_API_KEY` | https://platform.openai.com/api-keys | ano |
   | `TAVILY_API_KEY` | https://tavily.com (vyhledávání na webu) | ano |
   | `OPENROUTER_API_KEY` | https://openrouter.ai/keys (zapne filtr Jev, vyžaduje dobitý kredit) | ne |
   | `OPENAI_MODEL` | název modelu s podporou nástrojů a obrázků, výchozí `gpt-6-luna` | ne |

   Soubor `.env` je v `.gitignore` a do repozitáře se nikdy nenahrává.
3. Spusť aplikaci (`uv` při prvním spuštění samo nainstaluje závislosti):
   ```bash
   uv run main.py "Kolik je 1199 EUR v korunách?"
   ```
   Bez zadaného dotazu se aplikace zeptá sama.

Další příkazy:

```bash
uv run pytest               # automatické testy (rychlé, bez internetu a bez klíčů)
uv run test_queries.py      # všech 5 testovacích dotazů
uv run test_queries.py 4    # jen dotaz číslo 4
uv run try_tool.py convert_currency '{"amount": 100, "from_currency": "EUR", "to_currency": "CZK"}'
```

## Jak to funguje

```
Uživatel ─► LLM ──► potřebuje nástroj? ── ne ─► finální odpověď
              ▲            │ ano
              │            ▼
              │   validace argumentů (Pydantic)
              │            ▼
              │   spuštění nástrojů (souběžně)
              │            ▼
              └──── výsledky (nebo chyby) zpět do LLM
```

1. Uživatel zadá dotaz, ten se spolu s popisem nástrojů pošle LLM.
2. LLM buď odpoví rovnou, nebo požádá o jeden či více nástrojů.
3. Python každý požadavek zvaliduje a nástroj spustí. Nezávislé nástroje z jednoho kola běží **paralelně** (`ThreadPoolExecutor`).
4. Výsledky se vrátí LLM. Ten může požádat o další nástroje (kola na sebe navazují) nebo odpovědět.
5. Smyčka má limit 10 kol, aby se nezacyklila.

## Logování

Aplikace loguje přes modul `logging` (loggery `agent.main`, `agent.tools`, `agent.jev`) na dvě místa:

- **Konzole:** stručný průběh. Každé kolo vypíše `[step N] ... tool call(s) in parallel`, `[step N] -> nástroj(argumenty)` a `[step N] <- výsledek` (zkrácené na 150 a 200 znaků). Navíc se tu ukážou varování z Jev filtru, např. že se nepodařilo ohodnotit výsledek.
- **Soubor `agent.log`** (vedle kódu, rotuje po 1 MB, drží se 3 starší verze): vše podrobně, s časem, úrovní, názvem loggeru a **vláknem**, takže je vidět souběžné spouštění nástrojů. Obsahuje:
  - dotaz uživatele a finální odpověď,
  - délku každého volání LLM a počet vstupních a výstupních tokenů,
  - argumenty každého nástroje, jeho trvání a **úplný** výsledek (do konzole se zkracuje),
  - skóre Jev u každého výsledku hledání (`match`, `contradicts`, `not_confirmed`),
  - chyby jako `WARNING`: selhání nástroje a selhání volání OpenAI.

Soubor `agent.log` je v `.gitignore`, protože může obsahovat dotazy a výsledky hledání. API klíče se nelogují. Při ladění stačí otevřít `agent.log` a podívat se, co přesně LLM žádal a co nástroje vrátily, například při dotazu, na kterém se LLM zacyklí.

Samotné moduly nic nenastavují při importu, logování se zapne jen voláním `setup_logging()` ve `main.py` a `test_queries.py`. Automatické testy tedy žádný soubor nevytvářejí.

## Nástroje

| Nástroj | Co dělá | Argumenty | Externí služba |
|:--|:--|:--|:--|
| `web_search` | Hledání na webu, vrací výsledky (název, URL, úryvek) a URL obrázků | `query`, volitelně `country`, `max_results` (1–10), `requirements` (viz Jev filtr) | Tavily Search, volitelně Jev přes OpenRouter |
| `read_page` | Otevře stránku a vrátí jen části relevantní k dotazu (např. cenu konkrétní varianty) a obrázky ze stránky | `url`, `query` | Tavily Extract |
| `check_image` | Pošle obrázek z URL vision modelu a odpoví na otázku o něm (např. barva produktu) | `image_url`, `question` | OpenAI (vision) |
| `convert_currency` | Převod částky podle aktuálního kurzu | `amount` (> 0), `from_currency`, `to_currency` | Frankfurter (kurzy ECB, bez klíče) |

### Volitelný filtr Jev (rychlé řazení výsledků před LLM)

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) je model od TypeSafe AI, který **nepíše text, ale rozhoduje**: na zadané otázky vrací pravděpodobnosti, rychle (stovky ms) a levně. Hodí se tedy na hrubé třídění, které nemusí dělat drahý LLM.

Když LLM volá `web_search` s parametrem `requirements` (krátká ověřitelná tvrzení odvozená z dotazu, např. „Produkt je černý“, „Obchod je v Německu“), proběhne:

1. Tavily vrátí až 10 kandidátů.
2. Pro každý kandidát se souběžně zavolá Jev, který ke každému požadavku vybere jednu ze tří možností: **splněno**, **popřeno** (text říká opak) nebo **neuvedeno** (text o tom mlčí). Rozlišení „popřeno“ a „neuvedeno“ je důležité, protože úryvek často o vlastnosti (např. nové vs. repasované zboží) prostě nemluví.
3. Skóre výsledku je průměr přes požadavky (neuvedeno = půl bodu). LLM dostane jen nejlepší výsledky, každý se skóre `match` a seznamy `contradicts` a `not_confirmed`, tedy co ještě ověřit na stránce nástrojem `read_page`.

Jev ohodnotí kolem sedmi výsledků zhruba za půl sekundy. Filtr je **volitelný**: bez `OPENROUTER_API_KEY`, nebo když Jev selže, se `web_search` chová jako dřív a výsledek jen obsahuje poznámku, že požadavky nebyly použity. Výsledek, který Jev nestihl ohodnotit, se nezahodí, ale zařadí na konec.

Omezení: Jev posuzuje jen krátký úryvek a nevidí celou stránku. Na jemné rozdíly je citlivý na formulaci požadavku: u vágního „čip je Apple M5“ dostal model s čipem M5 **Pro** stejné skóre jako M5, u přesného „základní M5, ne M5 Pro nebo Max“ ho správně vyřadil. Systémový prompt proto LLM vede k přesným formulacím.

### Validace a chyby

- Argumenty od LLM přijdou jako text (JSON). Každý nástroj má vstupní **Pydantic model**, který zkontroluje typy, povinná pole, rozsahy a zakáže pole navíc. Ze stejného modelu se generuje schéma nástroje pro LLM, takže validace a popis nemohou být nekonzistentní.
- Funkce `execute_tool()` v `tools.py` **nikdy nevyhodí výjimku**. Neplatný JSON, neplatné argumenty, neznámý nástroj i selhání služby (síť, 404, vypršený čas) vrátí jako `{"error": "..."}`.
- Chybu dostane LLM jako výsledek nástroje a může na ni reagovat: opravit argumenty, zkusit jiný zdroj, nebo uživateli problém vysvětlit.
- Selhání samotného volání OpenAI (špatný klíč, překročený limit, výpadek sítě, vypršený čas) se vypíše jako srozumitelná hláška. Krátké výpadky OpenAI knihovna sama zopakuje (až 3×).
- Smyčka má limit 10 kol. V posledním kole se nástroje vypnou a LLM musí odpovědět s tím, co našel, a říct, co neověřil.

## Automatické testy

`uv run pytest` spustí 55 testů za zlomek sekundy. Nepotřebují internet ani API klíče, protože HTTP volání i OpenAI jsou nahrazené falešnými.

- **`tests/test_tools.py`** (nástroje): validace argumentů, chybové stavy (neplatný JSON, neznámý nástroj, chybějící klíč, 404, výpadek sítě), zkracování výstupu a tvar schémat pro LLM.
- **`tests/test_main.py`** (smyčka): odpověď bez nástroje, návrat výsledku nástroje do LLM, chyba nástroje, **souběžné** spuštění více nástrojů se zachováním `tool_call_id`, na sebe navazující kola, vynucená odpověď v posledním kole a převod chyb OpenAI na srozumitelné hlášky. LLM zde nahrazuje předem připravený scénář odpovědí.
- **`tests/test_jev_filter.py`** (filtr Jev): řazení a výpočet skóre, rozlišení „popřeno“ a „neuvedeno“, ořez na nejlepší výsledky, tvar dotazu na Jev, výsledek bez hodnocení se řadí na konec a návrat k původnímu pořadí, když Jev selže úplně.
- **`tests/test_logging.py`** (logování): co jde na konzoli a co do souboru, nezdvojování řádků při opakovaném nastavení, logování chyb nástrojů a Jev jako varování a logování kroků, plných výsledků, tokenů a finální odpovědi.

Pro kontrolu jsem do kódu úmyslně zanesl tři chyby (nástroje po jednom místo souběžně, poslední kolo nevypne nástroje, odstraněná validace částky) a každou chytil právě ten test, který ji má hlídat.

## Testovací dotazy a očekávané chování

Ukázkové dotazy proti skutečnému LLM se spouštějí pomocí `uv run test_queries.py`. Protože LLM není deterministický, test kontroluje jen to, **které nástroje se použily**, ne přesný text odpovědi.

| # | Scénář | Dotaz | Očekávané chování |
|:--|:--|:--|:--|
| 1 | Odpověď bez nástroje | „Jaké je hlavní město Francie?“ | LLM odpoví rovnou, žádný nástroj se nevolá |
| 2 | Jeden nástroj | „Kolik je 1199 EUR v korunách?“ | Jedno volání `convert_currency`, odpověď s přepočtenou částkou |
| 3 | Jiný nástroj | „Vyhledej na internetu, kdy byl představen čip Apple M5.“ | Volá se `web_search` (případně `read_page`), žádný převod měny |
| 4 | Více nástrojů | „Najdi nejlevnější černý Apple MacBook M5 v Německu, ověř podle obrázku, že je černý, a cenu převeď do CZK.“ | Řetězí se `web_search`, `read_page`, `check_image` a `convert_currency` ve více kolech |
| 5 | Chyba nástroje | „Je na tomto obrázku černý notebook? https://example.com/this-image-does-not-exist.jpg“ | `check_image` selže (obrázek nejde stáhnout), LLM chybu oznámí a nic si nevymyslí |

Příklad ze spuštění dotazu 4 ukázal, že se LLM umí z chyby zotavit: jedno volání `check_image` selhalo, model zkusil jiný obrázek a pokračoval.

Dotaz s paralelním voláním: „Kolik je 100 EUR, 100 GBP, 100 USD a 100 CHF v korunách?“ — LLM zavolá čtyřikrát `convert_currency` v jednom kole a všechna volání proběhnou najednou.

## Známá omezení

- **Výsledky se mění.** Hledání na webu i rozhodování LLM nejsou deterministické. Stejný dotaz 4 může vrátit jiného prodejce a jinou cenu. Ceny pocházejí ze stránek srovnávačů a jsou orientační (bez dopravy, mohou být zastaralé).
- **„Nejlevnější“ znamená nejlevnější z nalezených nabídek**, ne ověřené minimum celého trhu.
- **Obrázek nemusí odpovídat přesné variantě.** Srovnávače jako Geizhals sami uvádějí, že obrázky nemusí odpovídat skutečným produktům. Barva se proto bere hlavně z textu produktové stránky a obrázek ji doplňuje.
- Některé servery stahování obrázků blokují. Tento případ nástroj `check_image` vrátí jako chybu a LLM zkusí jiný obrázek.
- Kurzy Frankfurter jsou z ECB a o víkendu je k dispozici poslední pracovní den.
- Volání služeb (OpenAI, Tavily) stojí peníze nebo čerpají limit. Dotaz 4 obnáší řádově deset volání nástrojů a několik kol LLM, stojí zlomky dolaru a trvá kolem půl minuty.

## Implementované bonusy

- **Automatické testy** (`pytest`, 55 testů bez internetu)
- **Logování** (modul `logging`: konzole + rotující soubor `agent.log`)
- **Rychlý filtr před LLM** (Jev přes OpenRouter, volitelný)
- Validace dat přes **Pydantic**
- **Logování** jednotlivých kroků do konzole
- **Paralelní volání** nezávislých nástrojů
- Nástroje využívající **externí API** (Tavily, Frankfurter)
- Jednoduché **CLI** (dotaz jako argument nebo interaktivně)
