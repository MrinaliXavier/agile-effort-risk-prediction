"""
projects_config.py - the multi-project dataset definition.

Every scope decision lives here, fixed BEFORE any modelling result is seen:
  - which projects are included (rule: >= 1,000 resolved tickets with estimated
    Story Points, from src/scan_projects.py -> reports/project_scan.csv)
  - which Story Points field holds the team's estimate in each repository
  - Story Points outlier cap
  - which resolutions count as "work completed" (others are excluded from the
    delay label, e.g. Won't Fix / Duplicate tickets have a resolution date but
    no work was delivered)
"""

EARLY_WINDOW_DAYS = 5
SP_MAX = 100                     # tickets with Story Points > 100 are treated as data-entry errors
INCLUDE_DOC_PROJECTS = True      # set False to drop documentation-only projects (decide with supervisor)

# Estimated Story Points field(s) per repository, in priority order (first numeric value wins).
# From data/raw/.../jira_field_information.json, as found by scan_projects.py.
SP_FIELDS = {
    "Apache": ["customfield_12310293"],
    "IntelDAOS": ["customfield_10026", "customfield_10016"],
    "MongoDB": ["customfield_10555"],
    "Qt": ["customfield_10183"],
    "RedHat": ["customfield_12310243"],
    "Sonatype": ["customfield_10132"],
    "Spring": ["customfield_10142"],
}

# Sprint field per repository (field named exactly "Sprint"), found by src/scan_sprints.py.
SPRINT_FIELDS = {
    "Apache": ["customfield_12310921"],
    "IntelDAOS": ["customfield_10020"],
    "MongoDB": ["customfield_10557"],
    "Qt": ["customfield_10302"],
    "RedHat": ["customfield_12310940"],
    "Sonatype": ["customfield_11001"],
    "Spring": ["customfield_10480"],
}

# Projects with sprint information for >= 70% of completed tickets (reports/sprint_scan.csv).
# Rule fixed in advance: >= 8 such projects -> sprint spillover is the PRIMARY delay label (13 qualified).
# DAOS (34%) and NEXUS (62%) do not qualify.
SPILLOVER_PROJECTS = ["EVG", "XD", "MESOS", "WT", "DOCS", "DEVELOPER", "SERVER", "DROOLS",
                      "JBIDE", "RHDEVDOCS", "BXMSDOC", "KEYCLOAK", "QTBUG"]

# The 15 projects that passed the pre-set rule (reports/project_scan.csv).
PROJECTS = [
    {"repo": "MongoDB",   "key": "EVG",       "docs": False},
    {"repo": "IntelDAOS", "key": "DAOS",      "docs": False},
    {"repo": "Spring",    "key": "XD",        "docs": False},
    {"repo": "Apache",    "key": "MESOS",     "docs": False},
    {"repo": "MongoDB",   "key": "WT",        "docs": False},
    {"repo": "MongoDB",   "key": "DOCS",      "docs": True},
    {"repo": "RedHat",    "key": "DEVELOPER", "docs": False},
    {"repo": "Sonatype",  "key": "NEXUS",     "docs": False},
    {"repo": "MongoDB",   "key": "SERVER",    "docs": False},
    {"repo": "RedHat",    "key": "DROOLS",    "docs": False},
    {"repo": "RedHat",    "key": "JBIDE",     "docs": False},
    {"repo": "RedHat",    "key": "RHDEVDOCS", "docs": True},
    {"repo": "RedHat",    "key": "BXMSDOC",   "docs": True},
    {"repo": "RedHat",    "key": "KEYCLOAK",  "docs": False},
    {"repo": "Qt",        "key": "QTBUG",     "docs": False},
]

# A ticket counts as COMPLETED work only if it has a resolution date, a done-like
# status, and a resolution that is not one of these "no work delivered" outcomes.
DONE_STATUSES = {"done", "closed", "resolved", "fixed", "complete", "completed", "verified", "released"}
EXCLUDED_RESOLUTION_KEYWORDS = [
    "won't", "wont", "duplicate", "cannot reproduce", "can't reproduce", "invalid", "incomplete",
    "reject", "obsolete", "out of date", "abandon", "not a bug", "not a problem", "works as",
    "declined", "deferred", "later", "unresolved", "cancel", "explained", "gone away",
    "not an issue", "user error", "migrated",
]


def active_projects():
    return [p for p in PROJECTS if INCLUDE_DOC_PROJECTS or not p["docs"]]


def resolution_is_excluded(resolution_name):
    if resolution_name is None:
        return False
    name = str(resolution_name).lower()
    return any(k in name for k in EXCLUDED_RESOLUTION_KEYWORDS)
