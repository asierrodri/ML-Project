# Northern Lights Book Co. — Synthetic Retail Dataset (Calgary, AB) — 2019–2024

This dataset simulates a **medium-sized bookstore chain** in Calgary, Alberta with **3 stores and 1 warehouse**, covering **2019–2024**.
It is designed for **retail analytics, forecasting, payroll modeling, bank reconciliation practice, inventory analysis, and COVID-era trend studies**.

> **Note:** All people, transactions, and values are fully synthetic. Tax/benefit amounts are simplified placeholders for analysis and should not be used for compliance.

## Business Snapshot
- Name: Northern Lights Book Co.
- Structure: S-Corp with 2 shareholders
- Locations: YYC-DT (Downtown), YYC-NW (Northwest), YYC-SE (Southeast)
- Warehouse: YYC-WH (Calgary Warehouse)
- Years: 2019–2024
- Payroll: Permanent + Seasonal
- Loans: Government relief loan, Business line of credit
- Sales channels: In-store and online
- Currency: CAD
- Tax: GST 5% (AB)

## Files
- `bookstore_sales.csv` — Daily revenue by store/channel (incl. GST), 2019–2024
- `bookstore_inventory.csv` — Monthly WH→store SKU transfers & costs
- `bookstore_employees.csv` — Roster with roles, rates, dates
- `bookstore_payroll.csv` — Biweekly payroll (simplified deductions)
- `bookstore_checking.csv` — Checking account transactions (deposits, bills, payroll, loans)
- `bookstore_credit.csv` — Credit card charges (marketing, travel, inventory)
- `bookstore_loans.csv` — Quarterly loan balances & interest
- `bookstore_customers.csv` — Loyalty program customers

See `data_dictionary.csv` for full column descriptions.

## Known Simplifications
- Payroll deductions (CPP/EI/Income tax/benefits) are rough placeholders.
- Loan terms and government programs are simulated for realism but are not exact replicas of real program rules.
- Inventory depletion and COGS are not directly linked to sales at the SKU level (you can model this as an extension).

## Ideas for Analysis
- Estimate **COVID impact** on sales & staffing.
- Build **store forecasting models** using seasonality + shocks.
- Create a **cash flow model** from bank, sales, and payroll.
- Model **inventory allocation** and **stockout risk** by category.
- Segment **customers** by LTV and online ratio, then simulate **marketing lift**.

---

**Author:** Synthetic data generator notebook (Python). Feel free to use, fork, and extend.
