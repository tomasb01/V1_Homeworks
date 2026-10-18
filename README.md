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
| `test_queries.py` | 5 testovacích dotazů s automatickou kontrolou, které nástroje se použily |
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
   | `OPENAI_MODEL` | název modelu s podporou nástrojů a obrázků, výchozí `gpt-6-luna` | ne |

   Soubor `.env` je v `.gitignore` a do repozitáře se nikdy nenahrává.
3. Spusť aplikaci (`uv` při prvním spuštění samo nainstaluje závislosti):
   ```bash
   uv run main.py "Kolik je 1199 EUR v korunách?"
   ```
   Bez zadaného dotazu se aplikace zeptá sama.

Další příkazy:

```bash
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

Každý krok se vypisuje do konzole (`[step N] -> nástroj(argumenty)` a `[step N] <- výsledek`).

## Nástroje

| Nástroj | Co dělá | Argumenty | Externí služba |
|:--|:--|:--|:--|
| `web_search` | Hledání na webu, vrací výsledky (název, URL, úryvek) a URL obrázků | `query`, volitelně `country`, `max_results` (1–10) | Tavily Search |
| `read_page` | Otevře stránku a vrátí jen části relevantní k dotazu (např. cenu konkrétní varianty) a obrázky ze stránky | `url`, `query` | Tavily Extract |
| `check_image` | Pošle obrázek z URL vision modelu a odpoví na otázku o něm (např. barva produktu) | `image_url`, `question` | OpenAI (vision) |
| `convert_currency` | Převod částky podle aktuálního kurzu | `amount` (> 0), `from_currency`, `to_currency` | Frankfurter (kurzy ECB, bez klíče) |

### Validace a chyby

- Argumenty od LLM přijdou jako text (JSON). Každý nástroj má vstupní **Pydantic model**, který zkontroluje typy, povinná pole, rozsahy a zakáže pole navíc. Ze stejného modelu se generuje schéma nástroje pro LLM, takže validace a popis nemohou být nekonzistentní.
- Funkce `execute_tool()` v `tools.py` **nikdy nevyhodí výjimku**. Neplatný JSON, neplatné argumenty, neznámý nástroj i selhání služby (síť, 404, vypršený čas) vrátí jako `{"error": "..."}`.
- Chybu dostane LLM jako výsledek nástroje a může na ni reagovat: opravit argumenty, zkusit jiný zdroj, nebo uživateli problém vysvětlit.
- Selhání samotného volání OpenAI (špatný klíč, překročený limit, výpadek sítě, vypršený čas) se vypíše jako srozumitelná hláška. Krátké výpadky OpenAI knihovna sama zopakuje (až 3×).
- Smyčka má limit 10 kol. V posledním kole se nástroje vypnou a LLM musí odpovědět s tím, co našel, a říct, co neověřil.

## Testovací dotazy a očekávané chování

Spouštějí se pomocí `uv run test_queries.py`. Protože LLM není deterministický, test kontroluje jen to, **které nástroje se použily**, ne přesný text odpovědi.

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

- Validace dat přes **Pydantic**
- **Logování** jednotlivých kroků do konzole
- **Paralelní volání** nezávislých nástrojů
- Nástroje využívající **externí API** (Tavily, Frankfurter)
- Jednoduché **CLI** (dotaz jako argument nebo interaktivně)
