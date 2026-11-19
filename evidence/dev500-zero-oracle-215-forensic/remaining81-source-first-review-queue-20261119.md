# BIRD DEV500: Remaining 81 all-executable, all-wrong K8 questions — source-first review queue
2026-11-19. **Observational source-first triage, NOT an 81-case human-adjudicated mechanism report or an accuracy experiment.**
215 original zero-correct K8 cases contain 97 whose 8 original Qwen SQLs all executed but were wrong. The prior qualitative pilot covered 16; this lists the exact **81 residual questions across 11 databases**. All DEV500 source IDs and physical DBs were already exposed in historical research. Original model Qwen3.5-9B is fixed; no inference/training/Guarded change here.
**Review order**: question and original physical schema first → count/denominator entity, filter roles and table-key path → inspect original K8 model SQL → verify physical FK fanout/NULL semantics → only then inspect publisher reference SQL. Mark uncertain if no grounded answer. These snippets are truncated to 140 chars, not sufficient by themselves for full adjudication.
## By database
- card_games: 18
- formula_1: 13
- toxicology: 12
- codebase_community: 10
- european_football_2: 7
- thrombosis_prediction: 5
- debit_card_specializing: 5
- financial: 3
- superhero: 3
- student_club: 3
- california_schools: 2
## Mechanism hypotheses and explicit counterexamples
- #109 financial: user asks for women *clients*, while original SQL counts client×disp join rows. Multiple dispositions per client may inflate the result; no FK fanout measured yet.
- #1256/#1289 thrombosis_prediction: user asks for patients, candidate SQL counts laboratory/examination rows. Need record granularity and threshold definition before deciding DISTINCT.
- #1525 debit_card_specializing: user asks for a percentage of customers; the candidate SQL's transaction-row denominator may misalign with customer grain. Still unconfirmed.
- #1028 european_football_2: original COUNT(*) groups away-team wins by match; user asks for wins, so COUNT(*) may count exactly the right thing even with a JOIN. Never blanket-rewrite all joins.
- #1441 student_club: COUNT(*) after member JOIN major can be right if the relationship is many-members-to-one-major. Cannot infer a duplicate bug solely from JOIN.
- Original source-score-compatible positive controls selected for same-DB review: #168 financial, #1091 european_football_2, #1305 thrombosis_prediction. #1091 uses JOIN + COUNT(*) and was strict EX-correct, ruling out a global JOIN→DISTINCT rule.
**Decision: NO targeted Guarded rewrite.** The repeated pattern has not yet been physically verified across independent DBs. No baseline/treatment A/B, no EX gain and no rescues or regressions are claimed. HOLDOUT300 is not accessed.
## Complete 81-case source-first index
| ID | Database | Original question excerpt (140 chars) | Proposed entity/intent review | Different SQL strings in original 8 | State |
|---:|---|---|---|---:|---|
| 40 | california_schools | What is the telephone number for the school with the lowest average score in reading in Fresno Unified? | rank/top-k and grain | 7/8 | FK/grain unresolved |
| 50 | california_schools | What is the postal street address for the school with the 7th highest Math average? Indicate the school's name. | rank/top-k and grain | 7/8 | FK/grain unresolved |
| 340 | card_games | Which are the cards that have incredibly powerful foils. | projection/relationship/key | 2/8 | FK/grain unresolved |
| 347 | card_games | Find all cards illustrated by Stephen Daniel and describe the text of the ruling of these cards. State if these cards have missing or degrad | projection/relationship/key | 8/8 | FK/grain unresolved |
| 353 | card_games | List all the sets available in Italian translation. State the total number of cards per set. | COUNT: identify counted entity | 8/8 | FK/grain unresolved |
| 367 | card_games | Please provide the names of the artists who illustrated the card art in Phyrexian. | projection/relationship/key | 4/8 | FK/grain unresolved |
| 392 | card_games | Pick 3 cards with rarity of uncommon, list down name these cards according to ascending order of it's ruling date. | projection/relationship/key | 3/8 | FK/grain unresolved |
| 402 | card_games | What is the percentage of Story Spotlight cards that do not have a text box? | ratio/percent: numerator and denominator entity | 8/8 | FK/grain unresolved |
| 406 | card_games | List by ID all Creature-type cards with legal status. | projection/relationship/key | 5/8 | FK/grain unresolved |
| 412 | card_games | What is the foreign name of the card in French of type Creature, normal layout and black border color, by artist Matthew D. Wilson? | projection/relationship/key | 8/8 | FK/grain unresolved |
| 427 | card_games | What languages are available in the set known as Archenemy on the magic card market and having the code ARC? | projection/relationship/key | 3/8 | FK/grain unresolved |
| 428 | card_games | What is the name of set number 5 and its translation? | projection/relationship/key | 6/8 | FK/grain unresolved |
| 437 | card_games | Among black card borders, which card has full artwork? | projection/relationship/key | 6/8 | FK/grain unresolved |
| 462 | card_games | What's the Italian name of the set of cards with "Ancestor's Chosen" is in? | projection/relationship/key | 6/8 | FK/grain unresolved |
| 463 | card_games | How many translations are there for the set of cards with "Angel of Mercy" in it? | COUNT: identify counted entity | 6/8 | FK/grain unresolved |
| 473 | card_games | Is the set of cards with Adarkar Valkyrie only available outside the United States? | projection/relationship/key | 6/8 | FK/grain unresolved |
| 482 | card_games | What's the German type of the card "Ancestor's Chosen"? | projection/relationship/key | 4/8 | FK/grain unresolved |
| 484 | card_games | Please list the Italian names of the cards in the set Coldsnap with the highest converted mana cost. | rank/top-k and grain | 8/8 | FK/grain unresolved |
| 519 | card_games | What is the language of the "Battlebond" set? | projection/relationship/key | 5/8 | FK/grain unresolved |
| 527 | card_games | What are the rulings for the card named and designed by Kev Walker? List them in descending order of dates. | projection/relationship/key | 4/8 | FK/grain unresolved |
| 565 | codebase_community | User No.23853 gave a comment to a post at 9:08:18 on 2013/7/12, was that post well-finished? | projection/relationship/key | 8/8 | FK/grain unresolved |
| 571 | codebase_community | For the user No.24, how many times is the number of his/her posts compared to his/her votes? | COUNT: identify counted entity | 8/8 | FK/grain unresolved |
| 584 | codebase_community | Write all the comments left by users who edited the post titled 'Why square the difference instead of taking the absolute value in standard  | projection/relationship/key | 5/8 | FK/grain unresolved |
| 593 | codebase_community | How many users from New York have a teacher and supporter badge? | COUNT: identify counted entity | 7/8 | FK/grain unresolved |
| 620 | codebase_community | State the number of views of users who obtained the badge on 7/19/2010 7:39:08 PM. | COUNT: identify counted entity | 4/8 | FK/grain unresolved |
| 636 | codebase_community | How many negative comments did Neil McGuigan get in his posts? | COUNT: identify counted entity | 3/8 | FK/grain unresolved |
| 653 | codebase_community | What is the owner's display name of the most popular post? | rank/top-k and grain | 3/8 | FK/grain unresolved |
| 685 | codebase_community | Identify the total views on the post 'Computer Game Datasets'. Name the user who posted it last time. | projection/relationship/key | 8/8 | FK/grain unresolved |
| 694 | codebase_community | Provide the text of the latest 10 comments to the post with the title 'Analysing wind data with R' and the display name of the user who left | projection/relationship/key | 4/8 | FK/grain unresolved |
| 715 | codebase_community | In comments with score between 1 to 5, list down the display names of the users with 0 down votes. | projection/relationship/key | 5/8 | FK/grain unresolved |
| 1474 | debit_card_specializing | Which customers, paying in CZK, consumed the most gas in 2011? | rank/top-k and grain | 5/8 | FK/grain unresolved |
| 1494 | debit_card_specializing | What percentage of Slovakian gas stations are premium? | ratio/percent: numerator and denominator entity | 6/8 | FK/grain unresolved |
| 1502 | debit_card_specializing | Please list the chains of the gas stations with transactions in euro. | projection/relationship/key | 8/8 | FK/grain unresolved |
| 1525 | debit_card_specializing | What is the percentage of the customers who used EUR in 2012/8/25? | ratio/percent: numerator and denominator entity | 5/8 | FK/grain unresolved |
| 1529 | debit_card_specializing | What is the amount spent by customer 38508 at the gas stations? How much had the customer spent in January 2012? | projection/relationship/key | 8/8 | FK/grain unresolved |
| 1028 | european_football_2 | In Scotland Premier League, which away team won the most during the 2010 season? | rank/top-k and grain | 7/8 | FK/grain unresolved |
| 1069 | european_football_2 | Which football player has the shortest height? | rank/top-k and grain | 3/8 | FK/grain unresolved |
| 1080 | european_football_2 | Among the players whose preferred foot was the left foot when attacking, how many of them would remain in his position when the team attacke | COUNT: identify counted entity | 2/8 | FK/grain unresolved |
| 1092 | european_football_2 | Give the name of the league which had the most matches in the 2008/2009 season? | rank/top-k and grain | 4/8 | FK/grain unresolved |
| 1108 | european_football_2 | What was the build up play speed class for "Willem II" on 2011/2/22? | projection/relationship/key | 1/8 | FK/grain unresolved |
| 1143 | european_football_2 | What was the highest score of the home team in the Netherlands Eredivisie league? | rank/top-k and grain | 4/8 | FK/grain unresolved |
| 1145 | european_football_2 | Which top 4 leagues had the most games in the 2015-2016 season? | rank/top-k and grain | 4/8 | FK/grain unresolved |
| 109 | financial | How many clients opened their accounts in Jesenik branch were women? | COUNT: identify counted entity | 7/8 | FK/grain unresolved |
| 174 | financial | What year was account owner number 130 born? | COUNT: identify counted entity | 6/8 | FK/grain unresolved |
| 176 | financial | What is the amount of debt that client number 992 has, and how is this client doing with payments? | projection/relationship/key | 8/8 | FK/grain unresolved |
| 847 | formula_1 | What is the surname of the driver with the best lap time in race number 19 in the second qualifying period? | projection/relationship/key | 2/8 | FK/grain unresolved |
| 860 | formula_1 | For the driver who had the Q2 time as 0:01:40 in the qualifying race No. 355, what is his nationality? | projection/relationship/key | 3/8 | FK/grain unresolved |
| 861 | formula_1 | What is his number of the driver who finished 0:01:54 in the Q3 of qualifying race No.903? | COUNT: identify counted entity | 3/8 | FK/grain unresolved |
| 871 | formula_1 | For the driver who had the Q2 time as 0:01:15 in race No. 347, where is he from? | projection/relationship/key | 3/8 | FK/grain unresolved |
| 876 | formula_1 | For the race happened in 2015/11/29, how many drivers did not finish the race? | COUNT: identify counted entity | 6/8 | FK/grain unresolved |
| 877 | formula_1 | For all the drivers who finished the game in race No. 872, who is the youngest? | rank/top-k and grain | 5/8 | FK/grain unresolved |
| 883 | formula_1 | How many races were there in 2005? Name all the races in descending order. | COUNT: identify counted entity | 2/8 | FK/grain unresolved |
| 898 | formula_1 | How old is the youngest Japanese driver? What is his name? | rank/top-k and grain | 4/8 | FK/grain unresolved |
| 899 | formula_1 | List circuits which host 4 f1 races from year 1990 to 2000. | projection/relationship/key | 7/8 | FK/grain unresolved |
| 904 | formula_1 | State the race and year of race in which Michael Schumacher had his fastest lap. | projection/relationship/key | 8/8 | FK/grain unresolved |
| 922 | formula_1 | What time did the the 2010's Formula_1 race took place on the Abu Dhabi Circuit? | projection/relationship/key | 3/8 | FK/grain unresolved |
| 924 | formula_1 | Please list the exact dates on which a Formula_1 race took place on the Barcelona-Catalunya circuit. | projection/relationship/key | 2/8 | FK/grain unresolved |
| 984 | formula_1 | What is the website of the constructor who tallied the most total wins. | rank/top-k and grain | 3/8 | FK/grain unresolved |
| 1441 | student_club | How many members have education major in the College of Education & Human Services? | COUNT: identify counted entity | 2/8 | FK/grain unresolved |
| 1456 | student_club | List the full name of the top five members who spend the most money in the descending order of spending. | rank/top-k and grain | 3/8 | FK/grain unresolved |
| 1460 | student_club | Write the full name of the member who spent money for water, veggie tray and supplies and include the cost of it. | projection/relationship/key | 2/8 | FK/grain unresolved |
| 741 | superhero | What is the name of the superhero that has the most powers? | rank/top-k and grain | 4/8 | FK/grain unresolved |
| 766 | superhero | What is the hero's full name with the highest attribute in strength? | rank/top-k and grain | 1/8 | FK/grain unresolved |
| 842 | superhero | Calculate the average height of all neutral superheroes. | projection/relationship/key | 4/8 | FK/grain unresolved |
| 1213 | thrombosis_prediction | Name the ID of the patient who is born on the April 1st, 1982. Is his/her alkaliphophatase (ALP) within normal range? | projection/relationship/key | 6/8 | FK/grain unresolved |
| 1256 | thrombosis_prediction | How many patients with a abnormal C-reactive protein don't have their data recorded? | COUNT: identify counted entity | 2/8 | FK/grain unresolved |
| 1265 | thrombosis_prediction | How many patients have a normal level of anti-ribonuclear protein and have been admitted to the hospital? | COUNT: identify counted entity | 4/8 | FK/grain unresolved |
| 1269 | thrombosis_prediction | Please list the IDs of the patients who had the examination done after 1997/1/1 and had a normal anti-scl70. | projection/relationship/key | 8/8 | FK/grain unresolved |
| 1289 | thrombosis_prediction | For the patients who are diagnosed with SJS, how many of them have a normal level of total protein? | COUNT: identify counted entity | 3/8 | FK/grain unresolved |
| 219 | toxicology | What is the percentage of carcinogenic molecules in triple type bonds? | ratio/percent: numerator and denominator entity | 6/8 | FK/grain unresolved |
| 225 | toxicology | Please list top five molecules that have double bonds in alphabetical order. | rank/top-k and grain | 5/8 | FK/grain unresolved |
| 234 | toxicology | How many bonds which involved atom 12 does molecule TR009 have? | COUNT: identify counted entity | 4/8 | FK/grain unresolved |
| 237 | toxicology | Which molecule does the atom TR001_10 belong to? Please state whether this molecule is carcinogenic or not. | projection/relationship/key | 5/8 | FK/grain unresolved |
| 239 | toxicology | How many connections does the atom 19 have? | COUNT: identify counted entity | 5/8 | FK/grain unresolved |
| 248 | toxicology | What are the atoms of the triple bond with the molecule "TR041"? | projection/relationship/key | 7/8 | FK/grain unresolved |
| 267 | toxicology | List down the bond type for molecules from molecule id TR000 to TR050. | projection/relationship/key | 4/8 | FK/grain unresolved |
| 283 | toxicology | Identify whether the chemical compound that contains Calcium is carcinogenic. | projection/relationship/key | 4/8 | FK/grain unresolved |
| 304 | toxicology | List all carcinogenic molecules and their elements. | projection/relationship/key | 3/8 | FK/grain unresolved |
| 315 | toxicology | Among the molecules which contain "cl" element, which of them are carcinogenic? | projection/relationship/key | 3/8 | FK/grain unresolved |
| 328 | toxicology | List all the elements with double bond, consisted in molecule TR024. | projection/relationship/key | 7/8 | FK/grain unresolved |
| 329 | toxicology | Which carcinogenic molecule have the highest number of atoms consisted in it? | COUNT: identify counted entity | 6/8 | FK/grain unresolved |
## Evidence and costs
- Original source 32 sealed K8 Qwen3.5-9B generation caches and eight read-only bounded SQLite score workers; total 500 frozen DEV questions and 4,000 SQLs; 81×8=648 SQL candidates indexed, 0 new model calls or GPU usage.
- Full original K8 chosen SET-EX-compatible baseline 233/499 scorable; the NEW matched Guarded A/B on a fresh frozen source has NOT been run.
- In particular, this index is NOT an unblinded publisher-gold causal-label source and NOT an AppWorld/GAIA metric.
