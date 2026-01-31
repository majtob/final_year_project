# Data Entry Templates

Use these templates to manually collect data for the credit rating prediction project.

## Workflow

1. **Start with Tassnief** → Find all rated companies at https://tassnief.com
2. **Fill `companies_template.csv`** → Add each rated company with its latest rating
3. **Fill `ratings_template.csv`** → Add historical ratings (2021-2024) for each company
4. **Fill `kams_template.csv`** → Extract KAMs from annual reports
5. **Run conversion script** → Convert CSVs to `data_inventory.json`

## Templates

### companies_template.csv
Master list of companies in scope.

| Column | Description | Example |
|--------|-------------|---------|
| company_name | Full company name | Saudi Aramco |
| ticker | Tadawul ticker | 2222.SR |
| sector | Industry sector | Energy |
| tassnief_rating | Latest rating | A+ |
| rating_date | Date of latest rating | 2023-06-15 |
| outlook | Rating outlook | Stable / Positive / Negative |
| fiscal_year_applicable | Year the rating applies to | 2023 |
| source_url | Tassnief page URL | https://tassnief.com/... |
| notes | Any notes | |

### ratings_template.csv
Historical ratings for each company (one row per company per rating date).

| Column | Description |
|--------|-------------|
| ticker | Tadawul ticker |
| company_name | Company name |
| rating_date | Date rating was issued |
| rating | Credit rating (AAA, AA+, AA, AA-, A+, A, A-, BBB+, etc.) |
| outlook | Stable / Positive / Negative / Watch |
| rating_type | issuer / instrument |
| is_solicited | yes / no (was the rating requested by the company?) |
| rating_basis | full / pi (pi = public information only) |
| fiscal_year_applicable | Which fiscal year does this rating reflect |
| source_url | Link to Tassnief rating action |

**Rating Types Explained:**
- **Solicited (is_solicited=yes)**: Company requested and paid for the rating, provided non-public info
- **Unsolicited (is_solicited=no)**: Tassnief rated the company on their own initiative
- **Full basis**: Rating based on full information including non-public data
- **PI basis (pi)**: Rating based only on public information

**Important Notes for (pi) Ratings:**
- Use `outlook=N/A` for pi ratings (they don't carry an outlook)
- PI ratings are reviewed annually based on latest financial statements
- These are still valid ratings using parallel analytical procedures
- Company did not participate in the rating process

### kams_template.csv
Key Audit Matters extracted from annual reports.

| Column | Description |
|--------|-------------|
| ticker | Tadawul ticker |
| fiscal_year | Fiscal year of the annual report |
| kam_title | Title of the KAM |
| kam_category | Category (see below) |
| severity | low / medium / high |
| source_file | Annual report filename |
| source_page | Page number where KAM appears |
| notes | Any additional notes |

**KAM Categories:**
- `going_concern` - Going concern issues
- `impairment` - Asset impairment testing
- `revenue_recognition` - Revenue recognition complexity
- `related_party` - Related party transactions
- `litigation` - Legal/contingent liabilities
- `valuation` - Fair value measurements
- `inventory` - Inventory valuation
- `allowance` - Credit loss allowances (ECL)
- `other` - Other KAMs

## Rating Scale Reference

| Rating | Category | Numeric |
|--------|----------|---------|
| AAA | Investment Grade | 21 |
| AA+ | Investment Grade | 20 |
| AA | Investment Grade | 19 |
| AA- | Investment Grade | 18 |
| A+ | Investment Grade | 17 |
| A | Investment Grade | 16 |
| A- | Investment Grade | 15 |
| BBB+ | Investment Grade | 14 |
| BBB | Investment Grade | 13 |
| BBB- | Investment Grade | 12 |
| BB+ | Speculative | 11 |
| BB | Speculative | 10 |
| BB- | Speculative | 9 |
| B+ | Speculative | 8 |
| B | Speculative | 7 |
| B- | Speculative | 6 |
| CCC+ | Speculative | 5 |
| CCC | Speculative | 4 |
| CCC- | Speculative | 3 |
| CC | Speculative | 2 |
| C | Speculative | 1 |
| D | Default | 0 |

## Tips

1. **Focus on companies with multiple years of ratings** - This gives you more data points
2. **Prioritize companies with available English annual reports** - Easier KAM extraction
3. **Check both issuer and instrument ratings** - Some companies have sukuk ratings
4. **Note rating changes** - Upgrades/downgrades are interesting data points
