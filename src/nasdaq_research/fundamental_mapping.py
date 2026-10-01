"""Ordered, conservative US-GAAP mappings; first matching concept wins.

USD for all monetary fields; USD/shares for diluted EPS. Long-term debt is
noncurrent carrying value only, excluding current maturities and lease bundles.
CapEx is reported positive cash expenditure, not signed investing cash flow.
"""

# field: (ordered candidate concepts, unit, period basis)
CONCEPT_MAPPING = {
    "revenue": (("RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues"), "USD", "duration"),
    "cost_of_revenue": (("CostOfRevenue", "CostOfGoodsAndServicesSold"), "USD", "duration"),
    "gross_profit": (("GrossProfit",), "USD", "duration"),
    "operating_income": (("OperatingIncomeLoss",), "USD", "duration"),
    "net_income": (("NetIncomeLoss",), "USD", "duration"),
    "research_and_development": (("ResearchAndDevelopmentExpense",), "USD", "duration"),
    "selling_general_administrative": (("SellingGeneralAndAdministrativeExpense",), "USD", "duration"),
    "eps": (("EarningsPerShareDiluted",), "USD/shares", "duration"),
    "cash_and_cash_equivalents": (("CashAndCashEquivalentsAtCarryingValue",), "USD", "instant"),
    "total_assets": (("Assets",), "USD", "instant"),
    "total_liabilities": (("Liabilities",), "USD", "instant"),
    "stockholders_equity": (("StockholdersEquity",), "USD", "instant"),
    "current_assets": (("AssetsCurrent",), "USD", "instant"),
    "current_liabilities": (("LiabilitiesCurrent",), "USD", "instant"),
    "long_term_debt": (("LongTermDebtNoncurrent",), "USD", "instant"),
    "operating_cash_flow": (("NetCashProvidedByUsedInOperatingActivities",), "USD", "duration"),
    "capital_expenditures": (("PaymentsToAcquirePropertyPlantAndEquipment",), "USD", "duration"),
    "investing_cash_flow": (("NetCashProvidedByUsedInInvestingActivities",), "USD", "duration"),
    "financing_cash_flow": (("NetCashProvidedByUsedInFinancingActivities",), "USD", "duration"),
}
FINANCIAL_FIELDS = (*CONCEPT_MAPPING, "free_cash_flow")
