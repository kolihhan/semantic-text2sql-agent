# Blind source-and-schema adjudication: six BIRD DEV500 count-grain candidates
**Sealed 2026-11-19 before any new publisher reference-SQL examination for these six cases.** Derived solely from user question, original Qwen SQL and actual FK/PK metadata in GitHub Actions 44555827205. All source questions are already exposed DEV, so this is diagnostic, not independent validation.

## #109 financial
- Target counted entity: distinct woman **client** (`client.client_id`), with a customer–account relation through `disp.client_id`, `disp.account_id`, and account district. Schema confirms `disp.disp_id` PK, `client.client_id` PK.
- Original SQL counts joined `disp` rows; however original `COUNT(*)=25` and `COUNT(DISTINCT c.client_id)=25` on the actual filtered DB. Thus **no observed fanout overcount**. This is a direct counterexample to blanket DISTINCT. Root of the incorrect answer remains unproven (scope / branch-value mapping possible).

## #1028 european_football_2
- Target ranking entity: Scottish league **away team**, with numerator **won matches**. `Match.id` PK; `Match.away_team_api_id → Team.team_api_id` physical key in original SQL. Original `COUNT(*)=11` for Rangers and `COUNT(DISTINCT m.id)=11`.
- Counting each match is appropriate for *wins*; **not a player/team distinct-count problem**. Wrong result may instead be season scope: user says '2010 season' and model uses `2009/2010`; requires later reference/date interpretation. No force-correct claim.

## #1256 thrombosis_prediction
- Target counted entity: **patients** (`Patient.ID` PK), clinical abnormal `Laboratory.CRP` and missing patient information `Patient.Description` as one plausible reading.
- Laboratory PK is **(ID, Date)** and FK `Laboratory.ID → Patient.ID`: multiple lab dates per patient are structurally legal. Original `COUNT(*)=208`; substituting only `COUNT(DISTINCT P.ID)` yields **25**, a real 8.32× joined-row multiplicity. This is physical evidence of a candidate counting-grain issue, **not reference-EX proof**. Need to confirm whether missing 'data recorded' refers to Patient.Description or an exam/laboratory field.

## #1289 thrombosis_prediction
- Target counted entity: **patients** with SJS and normal total protein. Candidate joins `Examination.ID → Patient` to `Laboratory.ID → Patient` via ID. Examination has no declared PK, Laboratory PK (ID, Date), so potential fanout exists.
- Actual original filter produces **0**, and distinct-ID variant also **0**. Wrong predicate/normal-range or diagnosis scope may be blocking before grain. This is an explicit observed counterexample to assuming every multiple-record join needs DISTINCT to fix the task.

## #1441 student_club
- Target entity: **members** (`member.member_id` PK), filtered by `member.link_to_major → major.major_id`, and college/major language. This is a many-members-to-one-major declared FK, NOT an obvious join fanout.
- Actual original `COUNT(*)=8` and distinct member PK `=8`. SQL only filters `major.college`; user also asks for **education major**, so a major-name/department semantic filter may be missing. This is an unverified relationship/filter hypothesis, but proves DISTINCT does nothing on the original result.

## #1525 debit_card_specializing (ratio diagnostic, not a count substitution)
- Target is percentage of **customers**, not transactions, who used EUR on 2012-08-25. Physical PKs `customers.CustomerID` and `transactions_1k.TransactionID`. SQL uses transaction-row denominator `COUNT(*)` after joining on CustomerID, with currency in `customers.Currency`.
- Distinct customer as denominator may be appropriate, but whether a customer can have multiple transactions on that date (and how to define 'used EUR') requires explicit physical verification. This case was intentionally not mutated by the count-only probe, since rewriting a ratio needs separately validated numerator and denominator.

## Mechanism and promotion decision
- Frozen n=5 physically executed count-replacement probes across 4 DBs, only #1256 changes actual SQLite result (208→25); four remain unchanged. Additional ratio case #1525 annotated but not altered.
- No confirmed cross-database rescue or correct result. The single same-DB positive cannot license a generic Guarded fix. **Stop the rewrite.** Do not alter model/prompt/repair budget/evaluator or claim BIRD EX gain.
- Full 81-case source-first physical schema/SQL artifact exists in Actions 44555827205; 81-case human FK/answer-entity annotation is still incomplete. Revisit publisher SQL only AFTER this source-only annotation is fixed; holdout300 sealed.
