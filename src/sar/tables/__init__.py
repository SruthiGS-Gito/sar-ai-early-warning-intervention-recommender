"""Aggregated team tables for the app: department risk (ticket 203), retention funnel (ticket 207)
and student profiles (ticket 210). Each function returns one row per group, never per student;
groups smaller than sar.service.views.MIN_CELL are kept but their figures are blanked and marked
`suppressed`. scripts/build_team_tables.py writes the tables to data/processed/.
"""
