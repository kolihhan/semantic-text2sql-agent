# Text2SQL — BIRD DEV100 61 Wrong-Case Survey

Source: original Qwen3.5-9B guarded frozen100 experiment; raw 4 shard files from GitHub Actions run 37307108955. The original workflow did **not** finish as one clean end-to-end success; shard 3 is marked invalid. The 39/100 claim is a historical frozen artifact, not an independently replicated canonical score.

## Observed (not inferred)

- Correct 39/100; wrong 61/100.
- 52/61 physically execute but fail publisher execution match; 9/61 fail verification/execute.
- **42/52 executable-but-wrong** passed with **zero repairs**. Another 9 had 1 repair and 1 had 2 repairs.
- All 9 nonexecutable queries exhaust two repair attempts and end in `verification_failed`.
- Structural difference flags among the 52 executable-but-wrong (overlap!): join/aggregation 26, output-width 19, table references 18, literal values 17. Eight have none of these four coarse flags.
- These flags compare SQL syntax to publisher SQL; equivalent SQL may use different syntax, and incorrect SQL may have the same syntax. They are NOT causal roots.

## Selected question-to-SQL contradictions (qualitative examples)

| Case | Mechanism | Observation | Confidence |
|---|---|---|---|
| 169 | Growth-rate arithmetic | Question requests a growth rate; SQL projects numerator and denominator as separate output columns without calculating (new-old)/old. | HIGH |
| 325 | Wrong counting grain | Question counts carcinogenic molecules containing nitrogen; SQL COUNT(*) counts atom rows after joining molecules, potentially counting one molecule repeatedly. | HIGH |
| 298 | Wrong ratio grain | Question asks percentage of molecules; SQL numerator counts matching atoms while denominator counts molecules. | HIGH |
| 217 | Missing connectivity relation | Question asks atoms connected to atoms of molecule TR181; SQL selects its atoms directly, without tracing bonds/connectivity. | HIGH |
| 312 | Identifier/value column mix-up | Question provides a bond ID TR001_2_4; SQL filters b.bond_type using that identifier. | HIGH |
| 430 | Wrong selected entity | Question requests sets of cards and their IDs; SQL selects card IDs and orders by card name. | HIGH |
| 656 | Parent-child role reversal | Question requests the display name of the PARENT of highest-scoring child; SQL selects OwnerDisplayName from child rows with ParentId set. | HIGH |
| 96 | Age arithmetic | SQL computes birth year minus 1900 for an age-group breakdown, not current date minus birth year. | HIGH |
| 1438 | Link identifier used as display attribute | Question gives a link_to_major identifier; SQL filters major.college equal to that ID instead of grounding the link identifier. | HIGH |
| 884 | Wrong output entity and invalid dialect | Question requests race names; SQL projects driver names and calls EXTRACT(MONTH), unsupported in SQLite. | HIGH |
| 634 | Unbound aliases | SQL compares Harvey_ViewCount and Noah_ViewCount outside a subquery that only projects ViewCount and User, producing an invalid reference. | HIGH |
| 838 | Unsupported filter added | Question asks full name of superhero Alien; SQL adds publisher=Marvel even though publisher is not requested. | MEDIUM |
| 1061 | Weak first-name matching | Question asks first name Adam; SQL searches full player_name with substring LIKE %adam%, not the first-name token. | MEDIUM |
| 1197 | Identifier quoting / incorrect column | Final SQL contains Examination.Examination Date without quoting its spaced identifier; still fails verification after two repairs. | HIGH |
| 1226 | Hyphenated identifier parsed as subtraction | Unquoted T-CHO is interpreted as T minus CHO rather than the intended column name; fails verification after two repairs. | HIGH |

## Complete 61-case review index

Legend: **E** = failed verification/execution; **J** = SQL join/aggregation syntax differs from publisher reference; **P** = projection column count differs; **S** = referenced table set differs; **V** = literal values differ; **N** = no listed coarse difference. Letter flags are noncausal and may overlap.

| ID | Database | Executable | Repairs | Source difference flags | Question excerpt |
|---:|---|:---:|---:|---|---|
| 8 | california_schools | No | 2 | E | For the top 10 schools with the highest number of students eligible for free or reduced price meals, provide detailed informa |
| 28 | california_schools | Yes | 1 | N | Consider the average difference between K-12 enrollment and 15-17 enrollment of schools that are locally funded, list the nam |
| 38 | california_schools | Yes | 0 | V | What are the webpages for the Los Angeles County school that has between 2,000 and 3,000 test takers? |
| 60 | california_schools | No | 2 | E | For all partially virtual charter schools in San Joaquin County, provide a comprehensive analysis including their enrollment  |
| 69 | california_schools | No | 2 | E | For State Special Schools with a School Ownership Code starting with 3, show me their NCES district ID, school name, charter  |
| 77 | california_schools | Yes | 1 | SVJP | For schools in Los Angeles County that serve grades K-9, what are their poverty levels based on FRPM eligibility rates, and h |
| 84 | california_schools | Yes | 1 | SP | What are the two most common first names among the school administrators? Indicate the district to which they administer. |
| 93 | financial | Yes | 0 | N | How many male customers who are living in North Bohemia have average salary greater than 8000? |
| 96 | financial | Yes | 0 | SVJP | What is the demographic breakdown and financial profile of account owners who receive weekly statements, segmented by gender, |
| 111 | financial | No | 2 | E | What are the comprehensive statistics for accounts opened in Litomerice in 1996, including client demographics, transaction a |
| 121 | financial | No | 2 | E | For accounts opened in Prachatice district, provide a comprehensive financial profile including account details, owner demogr |
| 131 | financial | Yes | 0 | JP | Which district has highest active loan? |
| 169 | financial | Yes | 0 | VP | What was the growth rate of the total amount of loans across all accounts for a male client between 1996 and 1997? |
| 217 | toxicology | Yes | 0 | SJ | Identify all the atoms that are connected to the atoms of the TR181 molecule. |
| 287 | toxicology | Yes | 0 | J | Among all chemical compounds that contain molecule TR047, identify the percent that form a double-bond. |
| 298 | toxicology | Yes | 0 | N | Calculate the percentage of molecules containing carcinogenic compounds that element is hydrogen. |
| 300 | toxicology | Yes | 0 | V | What atoms comprise TR186? |
| 305 | toxicology | Yes | 1 | SJP | Name all bonds with single bond types and what atoms are connected to the molecules. |
| 312 | toxicology | Yes | 0 | SJ | What is the carcinogenic label for bond TR001_2_4? |
| 325 | toxicology | Yes | 0 | N | How many carcinogenic molecules that consisted of Nitrogen? |
| 348 | card_games | Yes | 0 | JP | Describe the information about rulings for card named 'Sublime Epiphany' with number 74s. |
| 421 | card_games | Yes | 0 | VP | Please list top three unknown power cards that have promotional types for arena league in alphabetical order. |
| 430 | card_games | Yes | 0 | SJP | Please list top two sets of cards with their IDs that have Italian-language cards and are located in the Shadowmoor block in  |
| 444 | card_games | Yes | 0 | N | Name the foreign name of the card that has boros watermark? List out the type of this card. |
| 455 | card_games | Yes | 0 | J | Which of the cards that are a promotional painting have multiple faces on the same card? Please list their names. |
| 476 | card_games | Yes | 0 | J | Please list the name of the cards in the set Coldsnap with the highest converted mana cost. |
| 513 | card_games | Yes | 0 | N | Among the sets whose expansion type is Commander, which set has the highest total number of cards including promotional and r |
| 515 | card_games | Yes | 0 | VJP | When was the oldest mythic card released and what are its legal play formats? |
| 546 | codebase_community | Yes | 0 | SJ | Please list the display names of all the users who owns a post that is well-finished. |
| 569 | codebase_community | Yes | 0 | VJ | Give the number of votes for the post about data visualization. |
| 586 | codebase_community | Yes | 0 | P | Which user added a bounty amount of 50 to the post title mentioning variance? |
| 634 | codebase_community | No | 2 | E | Among posts by Harvey Motulsky and Noah Snyder, which one has higher popularity? |
| 656 | codebase_community | Yes | 0 | SJ | Describe the display name of the parent ID for child post with the highest score. |
| 680 | codebase_community | Yes | 0 | SJ | What is the average score of Stephen Turner's posts? |
| 714 | codebase_community | Yes | 0 | P | List the comments commented by the user with a username of Harvey Motulsky. |
| 797 | superhero | Yes | 0 | P | Which superheroes have blue eyes with brown hair? |
| 812 | superhero | Yes | 0 | P | List down at least five full names of superheroes with blue eyes. |
| 838 | superhero | Yes | 0 | SVJ | Provide the full name of the superhero named Alien. |
| 884 | formula_1 | No | 2 | E | List the names of all races that occurred in the earliest recorded year and month. |
| 891 | formula_1 | Yes | 0 | VP | Name all drivers in the 2010 Singapore Grand Prix order by their position stands. |
| 952 | formula_1 | Yes | 0 | S | Which constructors have been ranked 1? |
| 973 | formula_1 | Yes | 0 | SJ | List out top 10 Spanish drivers who were born before 1982 and have the latest lap time. |
| 994 | formula_1 | Yes | 0 | J | Which constructor scored most points from Monaco Grand Prix between 1980 and 2010? List the score, name and nationality of th |
| 1017 | formula_1 | Yes | 0 | SV | Please list the location coordinates of the circuits whose lap record is 1:29.488. |
| 1061 | european_football_2 | Yes | 0 | V | How many players whose first names are Adam and weigh more than 170? |
| 1130 | european_football_2 | Yes | 1 | SJ | What are the short name of team who played safe while creating chance of passing? |
| 1133 | european_football_2 | Yes | 0 | V | How many football players born after the 1990s have the first name 'Aaron'? |
| 1142 | european_football_2 | Yes | 0 | SV | In the 2015–2016 season, how many games were played in the Italian Serie A league? |
| 1179 | thrombosis_prediction | Yes | 1 | VJP | For the patient who was diagnosed with SLE on 1994/2/19, what was his/her anti-Cardiolipin antibody concentration status on 1 |
| 1196 | thrombosis_prediction | Yes | 0 | P | What is the most common sign of patients with SLE disease? |
| 1197 | thrombosis_prediction | No | 2 | E | When was the medical information on patient number 48473 first documented, and what disease did she have? |
| 1226 | thrombosis_prediction | No | 2 | E | Who is the oldest patient with the highest total cholesterol (T-CHO). State the patient ID and T-CHO index. |
| 1243 | thrombosis_prediction | Yes | 1 | VJ | For all patients who are older than 55 years old, what is the percentage of female who has abnormal prothrombin time (PT)? |
| 1255 | thrombosis_prediction | Yes | 1 | SJ | For the patients with an abnormal Ig M level, what is the most common disease they are diagnosed with? |
| 1299 | thrombosis_prediction | Yes | 0 | N | Among the patients with the normal level of triglyceride, how many of them have other symptoms observed? |
| 1308 | thrombosis_prediction | Yes | 2 | SJ | Among the patients who have a normal platelet level, how many of them have other symptoms observed? |
| 1366 | student_club | Yes | 0 | P | List all the members who attended the event "October Meeting". |
| 1396 | student_club | Yes | 0 | JP | Provide the number of events attended by Luisa Guidi. What is her major? |
| 1438 | student_club | Yes | 0 | N | Please indicate the college of the person whose first name is Katy with the link to the major "rec1N0upiVLy5esTO". |
| 1485 | debit_card_specializing | Yes | 0 | V | How much more was customer 7 consuming in April 2013 than customer 5? |
| 1490 | debit_card_specializing | Yes | 1 | J | How many percent of LAM customer consumed more than 46.73? |

## Complete model SQL and stage statuses, one case at a time

### Case 8 — california_schools

**Question**: For the top 10 schools with the highest number of students eligible for free or reduced price meals, provide detailed information including their location, type, FRPM statistics, and SAT performance metrics.

**Execution**: failed; **repairs**: 2; **coarse flags**: E.

**Visible likely error** (HIGH, not adjudicated against reference SQL): Generated SQL names columns with spaces/parentheses; repairs do not reach SQLite-compilable SQL within two attempts.

**Final model SQL**:
```sql
SELECT 
    f."School Name" AS SchoolName,
    s.City,
    s.State,
    s.District AS DistrictName,
    f."District Type" AS DistrictType,
    f."School Type" AS SchoolType,
    f."County Name" AS CountyName,
    f."Enrollment (K-12)" AS Total_Enrollment_K12,
    f."FRPM Count (K-12)" AS FRPM_Count_K12,
    f."Percent (%) Eligible FRPM (K-12)" AS Percent_Eligible_FRPM_K12,
    s.AvgScrRead AS Avg_Score_Reading,
    s.AvgScrMath AS Avg_Score_Math,
    s.AvgScrWrite AS Avg_Score_Writing,
    s.NumGE1500 AS Num_Takers_GE1500
FROM 
    frpm f
JOIN 
    schools s ON f.CDSCode = s.CDSCode
JOIN 
    satscores sat ON s.CDSCode = sat.cds
WHERE 
    f."Academic Year" = '2013-14' AND f."2013-14 CALPADS Fall 1 Certification Status" = 'C'
ORDER BY 
    f."FRPM Count (K-12)" DESC
LIMIT 10;
```

### Case 28 — california_schools

**Question**: Consider the average difference between K-12 enrollment and 15-17 enrollment of schools that are locally funded, list the names and DOC type of schools which has a difference above this average.

**Execution**: passed; **repairs**: 1; **coarse flags**: N.

**Final model SQL**:
```sql
SELECT s.School, s.DOCType
FROM frpm f
JOIN schools s ON f.CDSCode = s.CDSCode
WHERE s.FundingType = 'Locally Funded'
AND (f."Enrollment (K-12)" - f."Enrollment (Ages 5-17)") > (
    SELECT AVG(f2."Enrollment (K-12)" - f2."Enrollment (Ages 5-17)")
    FROM frpm f2
    JOIN schools s2 ON f2.CDSCode = s2.CDSCode
    WHERE s2.FundingType = 'Locally Funded'
);
```

### Case 38 — california_schools

**Question**: What are the webpages for the Los Angeles County school that has between 2,000 and 3,000 test takers?

**Execution**: passed; **repairs**: 0; **coarse flags**: V.

**Visible likely error** (LOW, not adjudicated against reference SQL): SQL filters County = Los Angeles County; the encoded county value may differ (e.g. Los Angeles). Confirm actual DB values before assigning cause.

**Final model SQL**:
```sql
SELECT s.Website 
FROM schools s
JOIN satscores sc ON s.CDSCode = sc.cds
WHERE s.County = 'Los Angeles County'
  AND sc.NumTstTakr BETWEEN 2000 AND 3000;
```

### Case 60 — california_schools

**Question**: For all partially virtual charter schools in San Joaquin County, provide a comprehensive analysis including their enrollment statistics, poverty levels based on free and reduced-price meal eligibility, SAT performance metrics, and rank them by enrollment size within the county.

**Execution**: failed; **repairs**: 2; **coarse flags**: E.

**Visible likely error** (HIGH, not adjudicated against reference SQL): FRPM column names with spaces/parentheses remain improperly referenced after two repairs.

**Final model SQL**:
```sql
SELECT 
    f.[School Name] AS School_Name,
    f."Enrollment (K-12)" AS Enrollment_Stats,
    CASE 
        WHEN f."Percent (%) Eligible FRPM (K-12)" > 75 THEN 'High Poverty'
        WHEN f."Percent (%) Eligible FRPM (K-12)" BETWEEN 50 AND 75 THEN 'Medium-High Poverty'
        WHEN f."Percent (%) Eligible FRPM (K-12)" BETWEEN 25 AND 50 THEN 'Medium-Low Poverty'
        ELSE 'Low Poverty'
    END AS Poverty_Level_Category,
    ROUND(s.AvgScrRead, 2) AS SAT_Avg_Read,
    ROUND(s.AvgScrMath, 2) AS SAT_Avg_Math,
    ROUND(s.AvgScrWrite, 2) AS SAT_Avg_Write,
    s.NumGE1500 AS Num_SAT_1500_Plus,
    ROW_NUMBER() OVER (ORDER BY f."Enrollment (K-12)" DESC) AS Rank_By_Enrollment
FROM 
    frpm f
JOIN 
    schools s ON f.CDSCode = s.CDSCode
WHERE 
    f.[County Name] = 'San Joaquin'
    AND f."Charter School (Y/N)" = 'Y'
    AND s.Virtual = 'P'
ORDER BY 
    Rank_By_Enrollment;
```

### Case 69 — california_schools

**Question**: For State Special Schools with a School Ownership Code starting with 3, show me their NCES district ID, school name, charter status, current operational status, opening date, enrollment, poverty classification, SAT scores by subject, overall average SAT score, and enrollment ranking. Sort the results by enrollment from highest to lowest.

**Execution**: failed; **repairs**: 2; **coarse flags**: E.

**Visible likely error** (HIGH, not adjudicated against reference SQL): Final query uses invented normalized FRPM column identifiers not confirmed in physical schema; two repairs exhaust verification budget.

**Final model SQL**:
```sql
SELECT 
    s.NCESDist AS NCES_district_ID,
    s.School AS school_name,
    CASE WHEN f.Charter_School__Y_N = 'Y' THEN 'Charter' ELSE 'Not Charter' END AS charter_status,
    s.StatusType AS current_operational_status,
    s.OpenDate AS opening_date,
    f.Enrollment_K_12 AS enrollment,
    CASE 
        WHEN CAST(f.Percent_Eligible_FRPM_K_12 AS FLOAT) > 75 THEN 'High Poverty'
        WHEN CAST(f.Percent_Eligible_FRPM_K_12 AS FLOAT) > 50 THEN 'Medium Poverty'
        WHEN CAST(f.Percent_Eligible_FRPM_K_12 AS FLOAT) > 25 THEN 'Low Poverty'
        ELSE 'Very Low Poverty'
    END AS poverty_classification,
    sa.AvgScrRead AS sat_read_score,
    sa.AvgScrMath AS sat_math_score,
    sa.AvgScrWrite AS sat_write_score,
    ROUND(AVG(
        CASE 
            WHEN sa.AvgScrRead IS NOT NULL AND sa.AvgScrMath IS NOT NULL AND sa.AvgScrWrite IS NOT NULL 
            THEN (sa.AvgScrRead + sa.AvgScrMath + sa.AvgScrWrite) / 3.0
            ELSE NULL 
        END
    ), 2) AS overall_average_sat_score,
    RANK() OVER (ORDER BY f.Enrollment_K_12 DESC) AS enrollment_ranking
FROM 
    schools s
JOIN 
    frpm f ON s.CDSCode = f.CDSCode
LEFT JOIN 
    satscores sa ON s.CDSCode = sa.cds
WHERE 
    s.SOCType = 'State Special Schools'
    AND (s.SOC = '31' OR s.SOC LIKE '3%')
GROUP BY 
    s.NCESDist, s.School, s.StatusType, s.OpenDate, f.Charter_School__Y_N, sa.AvgScrRead, sa.AvgScrMath, sa.AvgScrWrite
ORDER BY 
    f.Enrollment_K_12 DESC;
```

### Case 77 — california_schools

**Question**: For schools in Los Angeles County that serve grades K-9, what are their poverty levels based on FRPM eligibility rates, and how do they perform on SAT tests? Include their charter status and rank them by FRPM percentage.

**Execution**: passed; **repairs**: 1; **coarse flags**: SVJP.

**Final model SQL**:
```sql
SELECT 
    s.CDSCode,
    s.School,
    s.County,
    AVG(f."Percent (%) Eligible FRPM (Ages 5-17)") AS Poverty_Level_Perc,
    CASE 
        WHEN AVG(f."Percent (%) Eligible FRPM (Ages 5-17)") >= 75 THEN 'High Poverty'
        WHEN AVG(f."Percent (%) Eligible FRPM (Ages 5-17)") BETWEEN 50 AND 74 THEN 'Medium Poverty'
        ELSE 'Low Poverty'
    END AS Poverty_Category,
    s.Charter AS Charter_Status,
    AVG(sc.AvgScrRead) AS Avg_SAT_Read,
    AVG(sc.AvgScrMath) AS Avg_SAT_Math,
    AVG(sc.AvgScrWrite) AS Avg_SAT_Write,
    AVG(sc.NumGE1500) AS Num_GE_1500
FROM schools s
JOIN frpm f ON s.CDSCode = f.CDSCode
LEFT JOIN satscores sc ON s.CDSCode = sc.cds
WHERE s.County = 'Los Angeles' 
    AND (LOWER(s.GSoffered) LIKE '%k%' OR LOWER(s.GSserved) LIKE '%k%') 
    AND (LOWER(s.GSoffered) LIKE '%9%' OR LOWER(s.GSserved) LIKE '%9%')
GROUP BY s.CDSCode, s.School, s.County, s.Charter
ORDER BY Poverty_Level_Perc DESC;
```

### Case 84 — california_schools

**Question**: What are the two most common first names among the school administrators? Indicate the district to which they administer.

**Execution**: passed; **repairs**: 1; **coarse flags**: SP.

**Final model SQL**:
```sql
SELECT AdmFName1 AS first_name, f."District Name" AS district_name, COUNT(*) AS frequency
FROM schools s
JOIN frpm f ON s.CDSCode = f.CDSCode
WHERE AdmFName1 IS NOT NULL AND AdmFName1 != ''
GROUP BY AdmFName1, f."District Name"
ORDER BY frequency DESC
LIMIT 2;
```

### Case 93 — financial

**Question**: How many male customers who are living in North Bohemia have average salary greater than 8000?

**Execution**: passed; **repairs**: 0; **coarse flags**: N.

**Final model SQL**:
```sql
SELECT COUNT(*) 
FROM client c
JOIN district d ON c.district_id = d.district_id
WHERE c.gender = 'M'
  AND d.A3 = 'North Bohemia'
  AND d.A11 > 8000;
```

### Case 96 — financial

**Question**: What is the demographic breakdown and financial profile of account owners who receive weekly statements, segmented by gender, age group, and region?

**Execution**: passed; **repairs**: 0; **coarse flags**: SVJP.

**Visible likely error** (HIGH, not adjudicated against reference SQL): SQL computes birth year minus 1900 for an age-group breakdown, not current date minus birth year.

**Final model SQL**:
```sql
SELECT 
    client.gender,
    CASE 
        WHEN strftime('%Y', client.birth_date) - strftime('%Y', '1900-01-01') < 30 THEN 'Young'
        WHEN (strftime('%Y', client.birth_date) - strftime('%Y', '1900-01-01')) >= 30 
             AND (strftime('%Y', client.birth_date) - strftime('%Y', '1900-01-01')) <= 50 THEN 'Middle-aged'
        ELSE 'Senior'
    END AS age_group,
    district.A3 AS region,
    ROUND(AVG(trans.amount * CASE WHEN trans.type = '+' THEN 1 ELSE -1 END), 2) AS avg_net_inflow
FROM account
JOIN client ON client.district_id = account.district_id
JOIN disp ON disp.client_id = client.client_id AND disp.account_id = account.account_id
JOIN district ON district.district_id = account.district_id
JOIN trans ON trans.account_id = account.account_id
WHERE account.frequency = 'POPLATEK TYDNE'
GROUP BY client.gender, age_group, region;
```

### Case 111 — financial

**Question**: What are the comprehensive statistics for accounts opened in Litomerice in 1996, including client demographics, transaction activity, loan information, and quarterly distribution of account openings?

**Execution**: failed; **repairs**: 2; **coarse flags**: E.

**Visible likely error** (MEDIUM, not adjudicated against reference SQL): Wide multi-table statistics query still fails SQLite compilation after two repairs; exact compiler token not preserved in summarized stages.

**Final model SQL**:
```sql
SELECT 
    client.client_id,
    client.gender,
    client.birth_date,
    district.A2 AS district_name,
    district.A3 AS region,
    COUNT(DISTINCT account.account_id) AS total_accounts_opened,
    COUNT(DISTINCT loan.loan_id) AS total_loans,
    SUM(CASE WHEN loan.status = 'A' THEN loan.amount ELSE 0 END) AS good_loan_amount,
    SUM(CASE WHEN loan.status = 'B' THEN loan.amount ELSE 0 END) AS bad_loan_amount,
    COUNT(DISTINCT order.order_id) AS total_orders,
    COUNT(DISTINCT trans.trans_id) AS total_transactions,
    CAST(SUM(trans.type * ABS(trans.amount)) AS INTEGER) AS approximate_total_transaction_value
FROM account
JOIN district ON account.district_id = district.district_id
LEFT JOIN client ON account.account_id IN (SELECT account_id FROM disp WHERE disp.client_id = client.client_id AND disp.account_id = account.account_id)
LEFT JOIN disp ON account.account_id = disp.account_id
LEFT JOIN loan ON account.account_id = loan.account_id
LEFT JOIN order AS o_order ON account.account_id = o_order.account_id
LEFT JOIN trans ON account.account_id = trans.account_id
WHERE district.A2 = 'Litomerice' 
    AND account.date LIKE '%1996%'
GROUP BY client.client_id, client.gender, client.birth_date, district.A2, district.A3;
```

### Case 121 — financial

**Question**: For accounts opened in Prachatice district, provide a comprehensive financial profile including account details, owner demographics, transaction statistics, loan information, customer categorization, and rank them by net balance from highest to lowest.

**Execution**: failed; **repairs**: 2; **coarse flags**: E.

**Visible likely error** (MEDIUM, not adjudicated against reference SQL): Account/customer multi-join analytics query remains noncompilable after two repairs; detailed compiler token unavailable.

**Final model SQL**:
```sql
SELECT 
    a.account_id,
    d.A2 AS district_name,
    a.date AS account_opened_date,
    a.frequency,
    c.gender,
    SUBSTR(c.birth_date, 1, 4) AS birth_year,
    COUNT(DISTINCT t.trans_id) AS transaction_count,
    SUM(CASE WHEN t.operation = 'PRIJEM' THEN t.amount ELSE -t.amount END) AS total_in_out_amount,
    COALESCE(MAX(t.balance), 0) AS last_balance_check,
    COUNT(l.loan_id) AS loan_count,
    MIN(l.date) AS first_loan_date,
    CASE 
        WHEN COUNT(l.loan_id) > 0 AND t.transaction_count > 10 THEN 'High Activity'
        WHEN COUNT(l.loan_id) > 0 THEN 'Loan Customer'
        WHEN t.transaction_count > 10 THEN 'Active Customer'
        ELSE 'Regular Customer'
    END AS customer_category,
    d.A4 AS inhabitants,
    d.A11 AS average_salary
FROM account a
JOIN district d ON a.district_id = d.district_id
LEFT JOIN disp d1 ON d1.account_id = a.account_id
JOIN client c ON d1.client_id = c.client_id
LEFT JOIN trans t ON t.account_id = a.account_id
LEFT JOIN loan l ON l.account_id = a.account_id
WHERE d.A2 LIKE 'Prachatice%'
GROUP BY a.account_id, d.A2, a.date, a.frequency, c.gender, SUBSTR(c.birth_date, 1, 4), d.A4, d.A11
ORDER BY total_in_out_amount DESC;
```

### Case 131 — financial

**Question**: Which district has highest active loan?

**Execution**: passed; **repairs**: 0; **coarse flags**: JP.

**Final model SQL**:
```sql
SELECT d.A2 AS district_name, MAX(l.amount) AS highest_active_loan
FROM loan l
JOIN account a ON l.account_id = a.account_id
JOIN district d ON a.district_id = d.district_id
WHERE l.status IN ('C', 'D')
GROUP BY d.A2
ORDER BY highest_active_loan DESC
LIMIT 1;
```

### Case 169 — financial

**Question**: What was the growth rate of the total amount of loans across all accounts for a male client between 1996 and 1997?

**Execution**: passed; **repairs**: 0; **coarse flags**: VP.

**Visible likely error** (HIGH, not adjudicated against reference SQL): Question requests a growth rate; SQL projects numerator and denominator as separate output columns without calculating (new-old)/old.

**Final model SQL**:
```sql
SELECT CAST(SUM(CASE WHEN loan.date LIKE '%1997%' THEN loan.amount ELSE 0 END) AS REAL) - SUM(CASE WHEN loan.date LIKE '%1996%' THEN loan.amount ELSE 0 END) AS numerator,
       SUM(CASE WHEN loan.date LIKE '%1996%' THEN loan.amount ELSE 0 END) AS denominator
FROM loan
JOIN account ON loan.account_id = account.account_id
JOIN disp ON disp.account_id = account.account_id
JOIN client ON disp.client_id = client.client_id
WHERE client.gender = 'M'
AND (loan.date LIKE '%1997%' OR loan.date LIKE '%1996%');
```

### Case 217 — toxicology

**Question**: Identify all the atoms that are connected to the atoms of the TR181 molecule.

**Execution**: passed; **repairs**: 0; **coarse flags**: SJ.

**Visible likely error** (HIGH, not adjudicated against reference SQL): Question asks atoms connected to atoms of molecule TR181; SQL selects its atoms directly, without tracing bonds/connectivity.

**Final model SQL**:
```sql
SELECT a.atom_id, a.element
FROM atom a
WHERE a.molecule_id = 'TR181';
```

### Case 287 — toxicology

**Question**: Among all chemical compounds that contain molecule TR047, identify the percent that form a double-bond.

**Execution**: passed; **repairs**: 0; **coarse flags**: J.

**Final model SQL**:
```sql
SELECT CAST(SUM(CASE WHEN bond.bond_type = '= ' THEN 1 ELSE 0 END) AS REAL) * 100.0 / COUNT(*) AS percent
FROM bond
WHERE bond.molecule_id = 'TR047';
```

### Case 298 — toxicology

**Question**: Calculate the percentage of molecules containing carcinogenic compounds that element is hydrogen.

**Execution**: passed; **repairs**: 0; **coarse flags**: N.

**Visible likely error** (HIGH, not adjudicated against reference SQL): Question asks percentage of molecules; SQL numerator counts matching atoms while denominator counts molecules.

**Final model SQL**:
```sql
SELECT COUNT(*) * 100.0 / (SELECT COUNT(*) FROM molecule WHERE label = '+') AS percentage
FROM atom
JOIN molecule ON atom.molecule_id = molecule.molecule_id
WHERE element = 'h' AND molecule.label = '+';
```

### Case 300 — toxicology

**Question**: What atoms comprise TR186?

**Execution**: passed; **repairs**: 0; **coarse flags**: V.

**Final model SQL**:
```sql
SELECT element FROM atom WHERE molecule_id = 186;
```

### Case 305 — toxicology

**Question**: Name all bonds with single bond types and what atoms are connected to the molecules.

**Execution**: passed; **repairs**: 1; **coarse flags**: SJP.

**Final model SQL**:
```sql
SELECT 
    c.bond_id,
    bt.molecule_id,
    m.label AS molecule_label,
    bt.bond_type,
    a1.element AS element1,
    a2.element AS element2
FROM bond bt
JOIN atom a1 ON bt.molecule_id = a1.molecule_id
JOIN atom a2 ON bt.molecule_id = a2.molecule_id
JOIN connected c ON bt.bond_id = c.bond_id
JOIN molecule m ON bt.molecule_id = m.molecule_id
WHERE bt.bond_type = '-'
    AND (c.atom_id = a1.atom_id OR c.atom_id2 = a1.atom_id)
    AND (c.atom_id = a2.atom_id OR c.atom_id2 = a2.atom_id);
```

### Case 312 — toxicology

**Question**: What is the carcinogenic label for bond TR001_2_4?

**Execution**: passed; **repairs**: 0; **coarse flags**: SJ.

**Visible likely error** (HIGH, not adjudicated against reference SQL): Question provides a bond ID TR001_2_4; SQL filters b.bond_type using that identifier.

**Final model SQL**:
```sql
SELECT m.label 
FROM molecule m
JOIN bond b ON m.molecule_id = b.molecule_id
WHERE b.bond_type = 'TR001_2_4';
```

### Case 325 — toxicology

**Question**: How many carcinogenic molecules that consisted of Nitrogen?

**Execution**: passed; **repairs**: 0; **coarse flags**: N.

**Visible likely error** (HIGH, not adjudicated against reference SQL): Question counts carcinogenic molecules containing nitrogen; SQL COUNT(*) counts atom rows after joining molecules, potentially counting one molecule repeatedly.

**Final model SQL**:
```sql
SELECT COUNT(*) 
FROM atom a
JOIN molecule m ON a.molecule_id = m.molecule_id
WHERE a.element = 'N' AND m.label = '+';
```

### Case 348 — card_games

**Question**: Describe the information about rulings for card named 'Sublime Epiphany' with number 74s.

**Execution**: passed; **repairs**: 0; **coarse flags**: JP.

**Final model SQL**:
```sql
SELECT r.date, r.text FROM rulings AS r WHERE r.uuid IN (SELECT c.uuid FROM cards AS c WHERE c.name = 'Sublime Epiphany' AND c.number = '74s');
```

### Case 421 — card_games

**Question**: Please list top three unknown power cards that have promotional types for arena league in alphabetical order.

**Execution**: passed; **repairs**: 0; **coarse flags**: VP.

**Final model SQL**:
```sql
SELECT name, promoTypes FROM cards WHERE isPromo = 1 AND promoTypes LIKE '%arenaleague%' AND (power IS NULL OR power = '*') ORDER BY asciiName LIMIT 3;
```

### Case 430 — card_games

**Question**: Please list top two sets of cards with their IDs that have Italian-language cards and are located in the Shadowmoor block in alphabetical order.

**Execution**: passed; **repairs**: 0; **coarse flags**: SJP.

**Visible likely error** (HIGH, not adjudicated against reference SQL): Question requests sets of cards and their IDs; SQL selects card IDs and orders by card name.

**Final model SQL**:
```sql
SELECT DISTINCT c.id
FROM cards c
JOIN sets s ON c.setCode = s.code
JOIN set_translations st ON s.code = st.setCode AND st.language = 'Italian'
WHERE s.block = 'Shadowmoor'
ORDER BY c.asciiName ASC
LIMIT 2;
```

### Case 444 — card_games

**Question**: Name the foreign name of the card that has boros watermark? List out the type of this card.

**Execution**: passed; **repairs**: 0; **coarse flags**: N.

**Final model SQL**:
```sql
SELECT fd.name AS foreign_name, fd.type AS card_type
FROM cards c
JOIN foreign_data fd ON c.uuid = fd.uuid
WHERE c.watermark = 'boros';
```

### Case 455 — card_games

**Question**: Which of the cards that are a promotional painting have multiple faces on the same card? Please list their names.

**Execution**: passed; **repairs**: 0; **coarse flags**: J.

**Final model SQL**:
```sql
SELECT name
FROM cards
WHERE isPromo = 1 AND side IS NOT NULL
GROUP BY id
HAVING COUNT(DISTINCT side) > 1;
```

### Case 476 — card_games

**Question**: Please list the name of the cards in the set Coldsnap with the highest converted mana cost.

**Execution**: passed; **repairs**: 0; **coarse flags**: J.

**Final model SQL**:
```sql
SELECT name
FROM cards
WHERE setCode = (SELECT code FROM sets WHERE name = 'Coldsnap')
ORDER BY convertedManaCost DESC;
```

### Case 513 — card_games

**Question**: Among the sets whose expansion type is Commander, which set has the highest total number of cards including promotional and related supplemental products but excluding Alchemy modifications? Indicate the id of the set.

**Execution**: passed; **repairs**: 0; **coarse flags**: N.

**Final model SQL**:
```sql
SELECT sets.id
FROM sets
WHERE sets.type = 'Commander'
ORDER BY COALESCE(sets.totalSetSize, 0) DESC
LIMIT 1;
```

### Case 515 — card_games

**Question**: When was the oldest mythic card released and what are its legal play formats?

**Execution**: passed; **repairs**: 0; **coarse flags**: VJP.

**Final model SQL**:
```sql
SELECT c.originalReleaseDate, l.format, l.status
FROM cards c
JOIN legalities l ON c.uuid = l.uuid
WHERE c.rarity = 'mythic'
ORDER BY c.originalReleaseDate ASC
LIMIT 1;
```

### Case 546 — codebase_community

**Question**: Please list the display names of all the users who owns a post that is well-finished.

**Execution**: passed; **repairs**: 0; **coarse flags**: SJ.

**Final model SQL**:
```sql
SELECT DISTINCT posts.OwnerDisplayName
FROM posts
WHERE posts.ClosedDate IS NOT NULL;
```

### Case 569 — codebase_community

**Question**: Give the number of votes for the post about data visualization.

**Execution**: passed; **repairs**: 0; **coarse flags**: VJ.

**Final model SQL**:
```sql
SELECT COUNT(*) as vote_count FROM votes WHERE PostId IN (SELECT Id FROM posts WHERE LOWER(Title) LIKE '%data%visualization%') AND VoteTypeId = 5;
```

### Case 586 — codebase_community

**Question**: Which user added a bounty amount of 50 to the post title mentioning variance?

**Execution**: passed; **repairs**: 0; **coarse flags**: P.

**Final model SQL**:
```sql
SELECT u.DisplayName
FROM users u
JOIN posts p ON u.Id = p.OwnerUserId
JOIN votes v ON p.Id = v.PostId AND u.Id = v.UserId
WHERE v.BountyAmount = 50
AND p.Title LIKE '%variance%';
```

### Case 634 — codebase_community

**Question**: Among posts by Harvey Motulsky and Noah Snyder, which one has higher popularity?

**Execution**: failed; **repairs**: 2; **coarse flags**: E.

**Visible likely error** (HIGH, not adjudicated against reference SQL): SQL compares Harvey_ViewCount and Noah_ViewCount outside a subquery that only projects ViewCount and User, producing an invalid reference.

**Final model SQL**:
```sql
SELECT 
    CASE 
        WHEN Harvey_ViewCount > Noah_ViewCount THEN 'Harvey Motulsky'
        ELSE 'Noah Snyder'
    END AS MorePopularUser
FROM (
    SELECT SUM(ViewCount) AS ViewCount, OwnerDisplayName AS User
    FROM posts
    WHERE OwnerDisplayName IN ('Harvey Motulsky', 'Noah Snyder')
    
    GROUP BY OwnerDisplayName
);
```

### Case 656 — codebase_community

**Question**: Describe the display name of the parent ID for child post with the highest score.

**Execution**: passed; **repairs**: 0; **coarse flags**: SJ.

**Visible likely error** (HIGH, not adjudicated against reference SQL): Question requests the display name of the PARENT of highest-scoring child; SQL selects OwnerDisplayName from child rows with ParentId set.

**Final model SQL**:
```sql
SELECT p1.OwnerDisplayName AS "Parent Display Name"
FROM posts p1
WHERE p1.ParentId IS NOT NULL
AND (p1.Id, p1.Score) = (
    SELECT Id, MAX(Score)
    FROM posts
    WHERE ParentId IS NOT NULL
    GROUP BY Id
    HAVING MAX(Score) <= 0
);
```

### Case 680 — codebase_community

**Question**: What is the average score of Stephen Turner's posts?

**Execution**: passed; **repairs**: 0; **coarse flags**: SJ.

**Final model SQL**:
```sql
SELECT AVG(posts.Score) 
FROM posts
WHERE posts.OwnerDisplayName = 'Stephen Turner';
```

### Case 714 — codebase_community

**Question**: List the comments commented by the user with a username of Harvey Motulsky.

**Execution**: passed; **repairs**: 0; **coarse flags**: P.

**Final model SQL**:
```sql
SELECT c.Text, c.CreationDate, c.Id 
FROM comments c 
JOIN users u ON c.UserId = u.Id 
WHERE u.DisplayName = 'Harvey Motulsky';
```

### Case 797 — superhero

**Question**: Which superheroes have blue eyes with brown hair?

**Execution**: passed; **repairs**: 0; **coarse flags**: P.

**Final model SQL**:
```sql
SELECT s.superhero_name, s.full_name
FROM superhero s
JOIN colour ce ON s.eye_colour_id = ce.id
JOIN colour ch ON s.hair_colour_id = ch.id
WHERE ce.colour = 'Blue' AND ch.colour = 'Brown';
```

### Case 812 — superhero

**Question**: List down at least five full names of superheroes with blue eyes.

**Execution**: passed; **repairs**: 0; **coarse flags**: P.

**Final model SQL**:
```sql
SELECT superhero.superhero_name, superhero.full_name
FROM superhero
JOIN colour ON superhero.eye_colour_id = colour.id
WHERE colour.colour = 'Blue'
LIMIT 5;
```

### Case 838 — superhero

**Question**: Provide the full name of the superhero named Alien.

**Execution**: passed; **repairs**: 0; **coarse flags**: SVJ.

**Visible likely error** (MEDIUM, not adjudicated against reference SQL): Question asks full name of superhero Alien; SQL adds publisher=Marvel even though publisher is not requested.

**Final model SQL**:
```sql
SELECT s.full_name 
FROM superhero s
JOIN publisher p ON s.publisher_id = p.id
WHERE s.superhero_name = 'Alien' AND p.publisher_name = 'Marvel';
```

### Case 884 — formula_1

**Question**: List the names of all races that occurred in the earliest recorded year and month.

**Execution**: failed; **repairs**: 2; **coarse flags**: E.

**Visible likely error** (HIGH, not adjudicated against reference SQL): Question requests race names; SQL projects driver names and calls EXTRACT(MONTH), unsupported in SQLite.

**Final model SQL**:
```sql
SELECT d.code || ' ' || d.surname AS driver_name
FROM drivers d
JOIN races r ON d.raceId = r.raceId
WHERE r.year = (SELECT MIN(year) FROM races)
   AND EXTRACT(MONTH FROM r.date) = (SELECT MIN(EXTRACT(MONTH FROM date)) FROM races);
```

### Case 891 — formula_1

**Question**: Name all drivers in the 2010 Singapore Grand Prix order by their position stands.

**Execution**: passed; **repairs**: 0; **coarse flags**: VP.

**Final model SQL**:
```sql
SELECT d.forename, d.surname, ds.position
FROM drivers d
JOIN driverStandings ds ON d.driverId = ds.driverId
JOIN races r ON ds.raceId = r.raceId
WHERE r.year = 2010 AND r.name = 'Singapore'
ORDER BY ds.position;
```

### Case 952 — formula_1

**Question**: Which constructors have been ranked 1?

**Execution**: passed; **repairs**: 0; **coarse flags**: S.

**Final model SQL**:
```sql
SELECT constructors.name
FROM constructors
JOIN constructorStandings ON constructors.constructorId = constructorStandings.constructorId
WHERE constructorStandings.position = 1;
```

### Case 973 — formula_1

**Question**: List out top 10 Spanish drivers who were born before 1982 and have the latest lap time.

**Execution**: passed; **repairs**: 0; **coarse flags**: SJ.

**Final model SQL**:
```sql
SELECT * FROM (
    SELECT 
        d.driverId,
        d.forename,
        d.surname,
        d.nationality,
        r.raceId,
        l.time AS lapTime
    FROM drivers d
    JOIN lapTimes l ON d.driverId = l.driverId
    JOIN races r ON l.raceId = r.raceId
    WHERE d.nationality = 'Spanish'
      AND strftime('%Y', d.dob) < '1982'
    ORDER BY l.time DESC
) AS top_drivers
LIMIT 10;
```

### Case 994 — formula_1

**Question**: Which constructor scored most points from Monaco Grand Prix between 1980 and 2010? List the score, name and nationality of this team.

**Execution**: passed; **repairs**: 0; **coarse flags**: J.

**Final model SQL**:
```sql
SELECT cr.points, c.name, c.nationality
FROM constructorResults cr
JOIN constructors c ON cr.constructorId = c.constructorId
JOIN races r ON cr.raceId = r.raceId
WHERE r.name = 'Monaco Grand Prix' AND r.year BETWEEN 1980 AND 2010
ORDER BY cr.points DESC
LIMIT 1;
```

### Case 1017 — formula_1

**Question**: Please list the location coordinates of the circuits whose lap record is 1:29.488.

**Execution**: passed; **repairs**: 0; **coarse flags**: SV.

**Final model SQL**:
```sql
SELECT lat, lng
FROM circuits
WHERE circuitId IN (
    SELECT DISTINCT c.circuitId
    FROM races r
    JOIN results res ON r.raceId = res.raceId
    JOIN circuits c ON r.circuitId = c.circuitId
    WHERE LOWER(c.name) LIKE '%1:29.488%'
       OR CAST(res.fastestLapTime AS TEXT) = '00:01:29.488'
);
```

### Case 1061 — european_football_2

**Question**: How many players whose first names are Adam and weigh more than 170?

**Execution**: passed; **repairs**: 0; **coarse flags**: V.

**Visible likely error** (MEDIUM, not adjudicated against reference SQL): Question asks first name Adam; SQL searches full player_name with substring LIKE %adam%, not the first-name token.

**Final model SQL**:
```sql
SELECT COUNT(*) FROM Player WHERE lower(player_name) LIKE '%adam%' AND weight > 170;
```

### Case 1130 — european_football_2

**Question**: What are the short name of team who played safe while creating chance of passing?

**Execution**: passed; **repairs**: 1; **coarse flags**: SJ.

**Final model SQL**:
```sql
SELECT DISTINCT T.team_short_name
FROM Team T
JOIN Match M ON (T.team_api_id = M.home_team_api_id OR T.team_api_id = M.away_team_api_id)
JOIN Team_Attributes TA ON (T.team_api_id = TA.team_api_id AND TA.date = M.date)
WHERE TA.chanceCreationPassingClass = 'Safe';
```

### Case 1133 — european_football_2

**Question**: How many football players born after the 1990s have the first name 'Aaron'?

**Execution**: passed; **repairs**: 0; **coarse flags**: V.

**Final model SQL**:
```sql
SELECT COUNT(*) 
FROM Player
WHERE player_name LIKE 'Aaron%'
  AND birthday > '1990-12-31';
```

### Case 1142 — european_football_2

**Question**: In the 2015–2016 season, how many games were played in the Italian Serie A league?

**Execution**: passed; **repairs**: 0; **coarse flags**: SV.

**Final model SQL**:
```sql
SELECT COUNT(*) 
FROM Match
WHERE Match.league_id IN (
    SELECT League.id 
    FROM League 
    JOIN Country ON League.country_id = Country.id 
    WHERE League.name = 'Serie A' AND Country.name = 'Italy'
)
  AND Match.season = 2015;
```

### Case 1179 — thrombosis_prediction

**Question**: For the patient who was diagnosed with SLE on 1994/2/19, what was his/her anti-Cardiolipin antibody concentration status on 1993/11/12?

**Execution**: passed; **repairs**: 1; **coarse flags**: VJP.

**Final model SQL**:
```sql
SELECT Examination.[aCL IgM] FROM Examination INNER JOIN Patient ON Examination.ID = Patient.ID WHERE Patient.Diagnosis = 'SLE' AND Patient.Description = '1994-02-19' AND Examination.[Examination Date] = '1993/11/12';
```

### Case 1196 — thrombosis_prediction

**Question**: What is the most common sign of patients with SLE disease?

**Execution**: passed; **repairs**: 0; **coarse flags**: P.

**Final model SQL**:
```sql
SELECT Symptoms, COUNT(*) AS frequency
FROM Examination
WHERE Diagnosis = 'SLE'
GROUP BY Symptoms
ORDER BY frequency DESC
LIMIT 1;
```

### Case 1197 — thrombosis_prediction

**Question**: When was the medical information on patient number 48473 first documented, and what disease did she have?

**Execution**: failed; **repairs**: 2; **coarse flags**: E.

**Visible likely error** (HIGH, not adjudicated against reference SQL): Final SQL contains Examination.Examination Date without quoting its spaced identifier; still fails verification after two repairs.

**Final model SQL**:
```sql
SELECT Patient.Description, Examination.Diagnosis
FROM Examination
JOIN Patient ON Examination.ID = Patient.ID
WHERE Patient.ID = 48473
ORDER BY Examination.Examination Date ASC
LIMIT 1;
```

### Case 1226 — thrombosis_prediction

**Question**: Who is the oldest patient with the highest total cholesterol (T-CHO). State the patient ID and T-CHO index.

**Execution**: failed; **repairs**: 2; **coarse flags**: E.

**Visible likely error** (HIGH, not adjudicated against reference SQL): Unquoted T-CHO is interpreted as T minus CHO rather than the intended column name; fails verification after two repairs.

**Final model SQL**:
```sql
SELECT p.ID, l.T-CHO
FROM Patient p
JOIN Laboratory l ON p.ID = l.ID
WHERE l.T-CHO = (
    SELECT MAX(T-CHO)
    FROM Laboratory
)
ORDER BY p.Birthday ASC, l.T-CHO DESC
LIMIT 1;
```

### Case 1243 — thrombosis_prediction

**Question**: For all patients who are older than 55 years old, what is the percentage of female who has abnormal prothrombin time (PT)?

**Execution**: passed; **repairs**: 1; **coarse flags**: VJ.

**Final model SQL**:
```sql
SELECT CAST(SUM(CASE WHEN PT >= 14 AND SEX = 'F' THEN 1 ELSE 0 END) AS REAL) * 100.0 / SUM(CASE WHEN PT >= 14 THEN 1 ELSE 0 END) AS percentage
FROM Laboratory L
JOIN Patient P ON L.ID = P.ID
WHERE (julianday('now') - julianday(P.Birthday)) / 365.25 > 55;
```

### Case 1255 — thrombosis_prediction

**Question**: For the patients with an abnormal Ig M level, what is the most common disease they are diagnosed with?

**Execution**: passed; **repairs**: 1; **coarse flags**: SJ.

**Final model SQL**:
```sql
SELECT E.Diagnosis
FROM Examination E
JOIN Patient P ON E.ID = P.ID
JOIN Laboratory L ON P.ID = L.ID
WHERE (L.IGM <= 40 OR L.IGM >= 400)
GROUP BY E.Diagnosis
ORDER BY COUNT(*) DESC
LIMIT 1;
```

### Case 1299 — thrombosis_prediction

**Question**: Among the patients with the normal level of triglyceride, how many of them have other symptoms observed?

**Execution**: passed; **repairs**: 0; **coarse flags**: N.

**Final model SQL**:
```sql
SELECT COUNT(*) 
FROM Examination E
JOIN Laboratory L ON E.ID = L.ID
WHERE L.TG < 200 AND E.Symptoms IS NOT NULL;
```

### Case 1308 — thrombosis_prediction

**Question**: Among the patients who have a normal platelet level, how many of them have other symptoms observed?

**Execution**: passed; **repairs**: 2; **coarse flags**: SJ.

**Final model SQL**:
```sql
SELECT COUNT(DISTINCT e.ID) AS patient_count
FROM Examination e
JOIN Patient p ON e.ID = p.ID
JOIN Laboratory l ON p.ID = l.ID
WHERE l.PLT BETWEEN 100 AND 400
AND e.Diagnosis IS NOT NULL;
```

### Case 1366 — student_club

**Question**: List all the members who attended the event "October Meeting".

**Execution**: passed; **repairs**: 0; **coarse flags**: P.

**Final model SQL**:
```sql
SELECT m.member_id, m.first_name, m.last_name
FROM member m
JOIN attendance a ON m.member_id = a.link_to_member
JOIN event e ON a.link_to_event = e.event_id
WHERE e.event_name = 'October Meeting';
```

### Case 1396 — student_club

**Question**: Provide the number of events attended by Luisa Guidi. What is her major?

**Execution**: passed; **repairs**: 0; **coarse flags**: JP.

**Final model SQL**:
```sql
SELECT member.first_name, member.last_name, major.major_name, COUNT(attendance.link_to_event) AS number_of_events_attended
FROM member
JOIN attendance ON member.member_id = attendance.link_to_member
JOIN major ON member.link_to_major = major.major_id
WHERE member.first_name = 'Luisa' AND member.last_name = 'Guidi'
GROUP BY member.member_id, major.major_id;
```

### Case 1438 — student_club

**Question**: Please indicate the college of the person whose first name is Katy with the link to the major "rec1N0upiVLy5esTO".

**Execution**: passed; **repairs**: 0; **coarse flags**: N.

**Visible likely error** (HIGH, not adjudicated against reference SQL): Question gives a link_to_major identifier; SQL filters major.college equal to that ID instead of grounding the link identifier.

**Final model SQL**:
```sql
SELECT major.college 
FROM member 
JOIN major ON member.link_to_major = major.major_id 
WHERE member.first_name = 'Katy' AND major.college = 'rec1N0upiVLy5esTO';
```

### Case 1485 — debit_card_specializing

**Question**: How much more was customer 7 consuming in April 2013 than customer 5?

**Execution**: passed; **repairs**: 0; **coarse flags**: V.

**Final model SQL**:
```sql
SELECT (SUM(CASE WHEN yearmonth.Date LIKE '%04%' AND substr(yearmonth.Date, 1, 7) = '2013' THEN yearmonth.Consumption ELSE 0 END)) - 
       (SUM(CASE WHEN yearmonth.Date LIKE '%04%' AND substr(yearmonth.Date, 1, 7) = '2013' THEN yearmonth.Consumption ELSE 0 END))
FROM yearmonth
WHERE CustomerID IN (5, 7)
```

### Case 1490 — debit_card_specializing

**Question**: How many percent of LAM customer consumed more than 46.73?

**Execution**: passed; **repairs**: 1; **coarse flags**: J.

**Final model SQL**:
```sql
SELECT COUNT(DISTINCT yearmonth.CustomerID) * 100.0 / (SELECT COUNT(DISTINCT CustomerID) FROM customers WHERE Segment = 'LAM') AS percentage FROM yearmonth JOIN customers ON yearmonth.CustomerID = customers.CustomerID WHERE customers.Segment = 'LAM' AND Consumption > 46.73
```

## Prioritization for the unchanged pipeline

1. Generator grounding: entity identity / relation paths / counting grain / proper numeric denominator; most (52) wrong SQL already execute.
2. Pre-generator schema catalog checks: physical column spelling, links, data types and FK/PK, especially 9 exhausted repair cases.
3. Preserve existing compiler verifier; do not call an executable query semantically correct.
4. Train-only verified interventions with the same Qwen base/model-call budget and matched physical EX. Do not tune on this historically seen DEV100.

The evidence does not support a claim that specific percentages of the 52 cases are caused by joins, table selection, literals, or output shapes. Only the observable syntax difference counts are measured.