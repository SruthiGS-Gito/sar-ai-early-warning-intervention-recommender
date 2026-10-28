"""Run the Streamlit app headless on synthetic data (streamlit.testing.v1.AppTest)."""

import re

import pandas as pd
import pytest

from sar.config import PROJECT_ROOT
from sar.explain.shap_explain import explain_rows
from sar.models.final import save_final
from sar.service.data import load_final_model, save_shap_cache

APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def _texts(at) -> str:
    parts = [e.value for e in at.info] + [e.value for e in at.warning] + [e.value for e in at.markdown]
    parts += [e.value for e in at.success] + [e.value for e in at.code]
    parts += [e.value for e in at.caption] + [str(e.value) for e in at.metric]
    return "\n".join(str(p) for p in parts if not str(p).startswith("<style>"))


@pytest.fixture
def empty_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SAR_MODELS_DIR", str(tmp_path / "models"))
    return tmp_path


@pytest.fixture
def synthetic_env(empty_env, matrix, small_final):
    processed = empty_env / "data" / "processed"
    processed.mkdir(parents=True)
    matrix.to_csv(processed / "day30_feature_matrix.csv", index=False)
    fitted, metrics = small_final
    save_final(fitted, {"seed": 42}, metrics, empty_env / "models" / "final")
    return empty_env


MAIN_PAGES = ["Overview", "Student", "Modules"]
PAGES = [*MAIN_PAGES, "Support", "About"]
DEVELOPER_TEXT = re.compile(r"SCRUM-|ticket \d|scripts/|\.py\b|docs/|models/|metrics\.json|last updated|\d+ rows")


def _go(at, page: str):
    """Switch pages the way the sidebar navigation does."""
    at.radio(key="nav_page").set_value(page).run()
    return at


def _tab(at, page: str):
    """The main area after switching to `page`."""
    return _go(at, page).main


def _visit(at, pages) -> str:
    """Text of `pages`, visited in turn; the app returns to the page it was on."""
    current = at.radio(key="nav_page").value
    text = "\n".join(_texts(_tab(at, p)) for p in pages)
    _go(at, current)
    return text


def _main_tab_texts(at) -> str:
    """Text of the sidebar and of every page except About."""
    return _texts(at.sidebar) + "\n" + _visit(at, [*MAIN_PAGES, "Support"])


def _open_training(at, presentation: str = "2013J"):
    at.toggle(key="show_training").set_value(True).run()
    at.selectbox(key="presentation").set_value(presentation).run()
    return at


def _about_text(at) -> str:
    return _visit(at, ["About"])


def test_app_runs_with_nothing_available(empty_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert [o.split()[-1] for o in at.radio(key="nav_page").options] == PAGES
    main, about = _main_tab_texts(at), _about_text(at)
    assert "not available yet" in main and "Data details" in main
    assert not DEVELOPER_TEXT.search(main), DEVELOPER_TEXT.search(main)
    assert "run_baseline.py" in about and "train_final.py" in about and "build_team_tables.py" in about
    assert "- Day 30 data: not loaded" in about and "- Model: not loaded" in about
    assert "SCRUM-" not in about and "ticket" not in about and "data/processed" not in about


def test_status_cards_describe_each_team_table(empty_env):
    from sar.service.data import OPTIONAL_TABLES

    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    main, about = _main_tab_texts(at), _about_text(at)
    for spec in OPTIONAL_TABLES.values():
        assert f"#### {spec.title}" in main and spec.description in main
        assert f"- {spec.title}: not loaded" in about
        assert spec.file not in about and not any(f"`{c}`" in about for c in spec.columns)
    assert main.count("Not available yet. Details are on the About page") == 3
    assert "### Risk bands by module" in main


def test_main_tabs_hold_no_developer_text_and_about_holds_the_details(tables_env):
    at = _go(AppTest.from_file(APP, default_timeout=120).run(), "Modules")
    at.toggle(key="show_3d").set_value(True).run()
    assert not at.exception
    main, about = _main_tab_texts(at), _about_text(at)
    assert not DEVELOPER_TEXT.search(main), DEVELOPER_TEXT.search(main)
    assert "24 inputs known by Day 30" in about and "isotonic calibration" in about.lower()
    assert "Trained on 2013B, 2013J, 2014B; on 2014J: ROC-AUC" in about
    bands = _tab(at, "About").dataframe[0].value
    assert list(bands["Band"]) == ["Low", "Medium", "High"] and "Capacity flagging" in about
    assert "rule-based" in about and "no interventions" in about
    assert "### Reading the Student page" in about and "not a personal probability" in about
    assert '"This dataset is released under CC-BY 4.0 license."' in about
    assert "doi: 10.1038/sdata.2017.171 (2017)." in about
    assert "Model built (UTC)" in about and "python scripts/build_team_tables.py" in about
    assert "- Department risk: loaded" in about and "SCRUM-" not in about and "data/processed" not in about
    assert "docs/" not in about


CARDS = ["Students in view", "Flagged for contact", "Precision", "At-risk reached"]
LIST_COLUMNS = ["Rank", "Student ID", "Module", "Risk band", "Day 30 snapshot", "Main reason", "Suggested support"]


def _listing(at):
    return [df for df in at.dataframe if "Main reason" in df.value.columns][0]


def _open_student(at, module=None, sid=None):
    """Choose a student the way a selected priority-list row does: module, student, then the Student tab."""
    if module is None:
        row = _listing(_go(at, "Overview")).value.iloc[0]
        module, sid = row["Module"], int(row["Student ID"])
    _go(at, "Student")
    at.selectbox(key="lookup_module").set_value(module).run()
    at.selectbox(key="lookup_student").set_value(sid).run()
    return module, sid


def test_priority_list_is_populated_on_first_load(synthetic_env, matrix, small_final):
    import json

    from sar.explain.shap_explain import explain_rows
    from sar.service import priority, views

    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert at.radio(key="nav_page").value == "Overview"
    tab = at.main
    cards = {m.label: m.value for m in tab.metric}
    assert list(cards) == CARDS and all(m.help for m in tab.metric)
    kpi = _texts(tab)
    assert kpi.count('class="kbar"><span') == 3 and kpi.count('class="kbar empty"') == 1   # one bar slot per card
    val = matrix[matrix["code_presentation"] == "2014J"]
    k = round(0.2 * len(val))
    assert cards["Students in view"] == f"{len(val):,}" and cards["Flagged for contact"] == f"{k:,}"
    assert "Highest-risk 20% of 2014J, all modules." in [c.value for c in tab.caption]
    frame = _listing(at)
    listing = frame.value
    assert list(listing.columns) == LIST_COLUMNS and len(listing) == k
    assert list(listing["Rank"]) == list(range(1, k + 1))
    assert json.loads(frame.proto.columns)["_index"]["hidden"] is True
    assert list(frame.proto.selection_mode) == [type(frame.proto).SelectionMode.Value("SINGLE_ROW")]
    scored = views.score_presentation(small_final[0], matrix, "2014J")
    top = scored[views.flag_capacity(scored, 0.2).flagged].sort_values("rank")
    assert list(listing["Student ID"]) == top["id_student"].astype(str).tolist()
    rows = top[views.KEY].merge(matrix, on=views.KEY)
    model = small_final[0]["model"]
    contrib = explain_rows(model, rows[list(model.feature_names_in_)])[1]
    details = priority.priority_details(rows[list(model.feature_names_in_)], contrib)
    assert list(listing["Main reason"]) == [priority.short_reason(r) for r in details["main_reason"]]
    assert list(listing["Suggested support"]) == details["suggested_support"].tolist()
    assert list(listing["Risk band"]) == top["band"].astype(str).tolist()
    specs = {s["layout"]["title"]["text"]: s for s in _chart_specs(at)}
    counts = priority.reason_counts(details["main_factor"])
    reasons = specs["Main reasons behind flags"]["data"][0]
    assert reasons["y"] == list(counts.index[::-1]) and sum(counts) == k
    assert reasons["text"] == [f"{n:,} ({n / k:.0%})" for n in counts.iloc[::-1]]
    donut = specs["Risk distribution"]["data"][0]
    bands = scored["band"].astype(str).value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    assert donut["values"] == bands.tolist() and donut["hole"] > 0
    assert donut["labels"][2] == f"High: {bands['High']:,} ({bands['High'] / len(val):.0%})"
    assert "Contact these" not in _texts(tab) and "-badge[" not in _texts(tab)
    assert any(b.proto.label == "Download the list (CSV)" for b in at.get("download_button"))
    assert "pseudonymised identifiers" in _texts(tab)


def test_student_view_shows_why_support_and_snapshot(synthetic_env, matrix):
    from sar.service.priority import snapshot

    at = AppTest.from_file(APP, default_timeout=120).run()
    tab = _tab(at, "Student")
    assert "Select a student in the Overview list" in _texts(tab)
    assert {b.label for b in tab.button} == {"Show an example High-risk student", "Show an example Low-risk student"}
    assert not any(b.proto.help for b in tab.button) and "drawn with a fixed seed" in _texts(tab)
    module, sid = _open_student(at)
    assert not at.exception
    tab = _tab(at, "Student")
    text = _texts(tab)
    assert f"### :material/person: Student {sid}" in text and f"Module **{module}** · Presentation 2014J" in text
    assert "This is a model prediction. It supports, and does not replace, professional judgment." in text
    from sar.service import views
    scored = views.score_presentation(load_final_model(synthetic_env / "models").value.fitted, matrix, "2014J")
    me = scored[(scored["code_module"] == module) & (scored["id_student"] == sid)].iloc[0]
    rate, n, scope = views.band_outcome(scored, module, str(me["band"]))
    ring = [s for s in _chart_specs(at) if s["data"][0].get("hole") == 0.78][0]
    assert ring["layout"]["annotations"][0]["text"] == f"<b>{rate:.0%}</b>"
    assert ring["data"][0]["values"] == pytest.approx([rate, 1 - rate])
    assert "**Observed outcome for this band**" in text and f"(n = {n:,})" in text
    where = f"in module {module}" if scope == module else "across all modules"
    assert f"of 2014J students in the {me['band']} band {where} went on to fail or withdraw" in text
    assert f"Calibrated model score {me['score']:.2f}, calibrated across all modules." in text
    assert "Estimated risk" not in text
    assert [t.label for t in tab.tabs] == ["Overview", "Day 30 indicators", "Why this score", "Support"]
    assert re.search(r'class="pill pill-(high|medium|low)">(High|Medium|Low) risk</span>', text)
    assert re.search(rf"ranked \d+ of [\d,]+ in module", text)
    why = [s for s in _chart_specs(at) if s["layout"].get("title", {}).get("text") == "What moves this score"][0]
    traces = {t["name"]: t for t in why["data"]}
    assert set(traces) == {"Raises risk", "Lowers risk"}
    assert why["layout"]["xaxis"]["title"]["text"].startswith("Change in the raw forest score")
    assert traces["Raises risk"]["marker"]["color"] == "#ff5d73" and all(x > 0 for x in traces["Raises risk"]["x"])
    assert all(x <= 0 for x in traces["Lowers risk"]["x"])
    shown = [p for t in why["data"] for p in t["customdata"]]
    assert len(traces["Raises risk"]["x"]) <= 4 and len(traces["Lowers risk"]["x"]) <= 2
    assert 1 <= len(shown) <= 6 and not any(p.startswith("being on module") for p in shown)
    verdict = [m.value for m in tab.markdown if m.value.startswith(("The risk score", "No activity"))][0]
    named = re.findall(r"(?:raised most by|lowered most by) (.+?)\.(?: |$)", verdict)
    assert all(p in shown for chunk in named for p in re.split(r", | and ", chunk))
    assert ("It is lowered most by" in verdict) == bool(traces["Lowers risk"]["x"])
    assert all(len(line) <= 30 for t in why["data"] for y in t["y"] for line in y.split("<br>"))
    assert "Red bars raise the risk score and teal bars lower it" in text
    assert re.search(r"(The risk score is raised most by|No activity, assessment or study-load input raises)", text)
    assert text.count("descriptive, not evidence of effectiveness") == 1 and "OULAD records no interventions" in text
    assert re.search(r"Similar earlier students: the [\d,]+ students (in module \w+|across all modules) from the "
                     r"training presentations", text)
    assert "similar earlier students, " not in text                       # no longer on the primary card
    assert "Too few similar students" not in text and "linked to" not in text
    snap = snapshot(matrix, "2014J", module, sid, {"active_days": "Active days", "n_missed": "Assessments missed"})
    metrics = {m.label: m for m in tab.metric}
    for f, r in snap.iterrows():
        m = metrics[r["label"]]
        diff = r["difference"]
        assert m.value == f"{r['value']:,.0f}" and not m.delta
        word = "Same as" if diff == 0 else "Above" if diff > 0 else "Below"
        assert f'pill-info">{word}</span>&nbsp; module median {r["median"]:,.0f}' in text
    assert "provisional" not in text.lower()


def test_flagging_rule_and_three_d_view_from_the_sidebar(synthetic_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert at.radio(key="flag_rule").value == "Capacity ranking" and at.slider(key="capacity").value == 20
    at.radio(key="flag_rule").set_value("Wide net (fixed cutoff)").run()
    assert not at.exception and "Wide net: calibrated risk score of at least" in _texts(at)
    _go(at, "Modules")
    assert not at.toggle(key="show_3d").value
    at.toggle(key="show_3d").set_value(True).run()
    assert not at.exception


def test_module_filter_limits_the_list_and_the_cards(synthetic_env, matrix):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="module_filter").set_value("AAA").run()
    assert not at.exception
    listing = _listing(at).value
    assert set(listing["Module"]) <= {"AAA"}
    n = int(((matrix["code_presentation"] == "2014J") & (matrix["code_module"] == "AAA")).sum())
    assert {m.label: m.value for m in at.metric}["Students in view"] == f"{n:,}"
    assert "Highest-risk 20% of 2014J, module AAA." in _texts(at)


def test_lookup_uses_precomputed_explanations_when_present(synthetic_env, matrix):
    models = synthetic_env / "models"
    loaded = load_final_model(models).value
    fitted = loaded.fitted
    part = matrix[matrix["code_presentation"] == "2014J"]
    base, contrib, raw = explain_rows(fitted["model"], part[fitted["features"]])
    save_shap_cache(models, "2014J", part[["code_module", "code_presentation", "id_student"]], contrib, raw, base,
                    loaded.config)
    at = AppTest.from_file(APP, default_timeout=120).run()
    _open_student(at)
    assert not at.exception
    assert "Background characteristics are not model inputs." in _texts(at)
    listing = _listing(_go(at, "Overview")).value
    assert not (listing["Main reason"] == "Open the student to see").any()


def test_training_presentation_is_labelled(synthetic_env):
    at = _open_training(AppTest.from_file(APP, default_timeout=120).run())
    assert not at.exception
    assert "training presentation" in _texts(at)


@pytest.fixture
def tables_env(synthetic_env, matrix, small_final):
    """Synthetic environment with the three team tables built by sar.tables."""
    from conftest import synthetic_registrations

    from sar.data.split import make_split
    from sar.tables import department, funnel, profiles

    processed = synthetic_env / "data" / "processed"
    department.department_risk(small_final[0], matrix).to_csv(processed / "department_risk.csv", index=False)
    funnel.retention_funnel(*synthetic_registrations(matrix), matrix).to_csv(
        processed / "retention_funnel.csv", index=False)
    train, val = make_split(matrix)
    table = profiles.student_profiles(profiles.fit_profiles(train), train, val)
    table.to_csv(processed / "student_profiles.csv", index=False)
    return synthetic_env, table


def _outside_expanders(node, kind):
    """Elements of type `kind` that are not inside an expander."""
    for child in getattr(node, "children", {}).values():
        if child.type == kind:
            yield child
        elif child.type != "expander":
            yield from _outside_expanders(child, kind)


def test_cohort_insights_shows_cards_charts_and_collapsed_tables(tables_env):
    _, prof = tables_env
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    tab = _tab(at, "Modules")
    text = _texts(tab)
    assert "not available yet" not in text.lower()
    for heading in ["### Risk bands by module", "### Student profiles", "### Retention funnel"]:
        assert heading in text
    dept = pd.read_csv(tables_env[0] / "data" / "processed" / "department_risk.csv")
    dept = dept[dept["code_presentation"] == "2014J"]
    modules = sorted(set(dept["code_module"]) - {"All modules"})
    cards = {m.label: m.value for m in tab.metric}
    assert list(cards) == ["All modules", *[f"Module {m}" for m in modules]]
    for _, row in dept.iterrows():
        expected = "n < 20" if row["suppressed"] else f"{row['at_risk_rate']:.0%}"
        label = "All modules" if row["code_module"] == "All modules" else f"Module {row['code_module']}"
        assert cards[label] == expected and f"n = {int(row['students']):,} students" in text
    assert text.count("Observed at-risk rate") >= len(cards)
    assert text.count('class="bandbar"') == int((~dept["suppressed"].astype(bool)).sum())
    assert not list(_outside_expanders(tab, "dataframe"))                   # tables only in collapsed sections
    assert [e.label for e in tab.expander] == ["What each profile means", "Data tables", "Model reliability",
                                               "Optional 3D view"]
    assert "Retrospective: the last two stages use final outcomes, which are not model inputs." in text
    validation = prof[prof["scope"] == "validation"]
    for _, row in validation.iterrows():
        assert f"**{row['profile']}**: {row['description']}" in text
    about = _about_text(at)
    assert "Counts under 20 are hidden" in about and "Groups under 20 students" in about
    assert "without background characteristics" in about and "{" not in text


def test_team_table_selectors_and_training_note(tables_env):
    at = _go(_open_training(AppTest.from_file(APP, default_timeout=120).run()), "Modules")
    assert not at.exception and "Training presentation: the model learned" in _texts(at)
    assert "Shown for the training presentations together" in _texts(at)
    at.selectbox(key="funnel_module").set_value("AAA").run()
    assert not at.exception


def test_student_level_team_table_is_refused(synthetic_env):
    processed = synthetic_env / "data" / "processed"
    pd.DataFrame({"code_module": ["AAA"], "id_student": [1]}).to_csv(processed / "department_risk.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    main, about = _main_tab_texts(at), _about_text(at)
    assert "the file contains student-level rows" in main
    assert "- Department risk: not loaded" in about and "- Retention funnel: not loaded" in about


def test_missing_columns_are_named_in_the_data_details(synthetic_env):
    processed = synthetic_env / "data" / "processed"
    pd.DataFrame({"stage": ["Registered"]}).to_csv(processed / "retention_funnel.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert "the file does not have the expected columns" in _main_tab_texts(at)
    assert "- Retention funnel: not loaded" in _about_text(at)


def test_funnel_shows_a_stage_that_does_not_apply_as_not_applicable(synthetic_env, matrix):
    from conftest import synthetic_registrations

    from sar.tables import funnel

    no_due = matrix.assign(n_assessments_due=matrix["n_assessments_due"].where(matrix["code_module"] != "GGG", 0))
    table = funnel.retention_funnel(*synthetic_registrations(matrix), no_due)
    table.to_csv(synthetic_env / "data" / "processed" / "retention_funnel.csv", index=False)
    at = _go(AppTest.from_file(APP, default_timeout=120).run(), "Modules")
    at.selectbox(key="funnel_module").set_value("GGG").run()
    assert not at.exception
    text = _texts(at)
    assert "submission stage is not applicable" in _about_text(at)
    carried = table[(table["code_module"] == "GGG") & (table["code_presentation"] == "2014J")
                    & ~table["applicable"]]["students"].sum()
    assert f"{carried:,} students in GGG are carried forward on the submission check" in text
    funnel_view = [df.value for df in at.dataframe if "Stage" in df.value.columns][0]
    assert "not applicable" in set(funnel_view["Students"])
    assert funnel_view["Stage"].str.endswith("(not applicable)").sum() == 1


def test_band_views_default_to_the_validation_presentation(tables_env):
    at = _go(AppTest.from_file(APP, default_timeout=120).run(), "Modules")
    assert not at.exception
    text = _texts(at)
    assert at.selectbox(key="presentation").options == ["2014J (validation)"]
    assert at.selectbox(key="presentation").value == "2014J"
    assert not _go(at, "Overview").toggle(key="show_training").value
    _go(at, "Modules")
    assert "Training presentations have in-sample scores" not in _texts(at.sidebar)
    assert "This is a training presentation" not in _texts(at.sidebar) and "Validated on" not in _texts(at.sidebar)
    dept = [df.value for df in at.dataframe if "High band" in df.value.columns][0]
    assert "Presentation" not in dept.columns and "Scores" not in dept.columns
    assert "**Risk bands by module, 2014J** (validation (2014J) scores)" in text


def test_training_presentations_appear_only_after_opting_in(tables_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.toggle(key="show_training").set_value(True).run()
    assert not at.exception
    assert _texts(at.sidebar).count("Training presentations have in-sample scores") == 1
    assert at.selectbox(key="presentation").options == [f"{p} (training, in-sample)" for p in
                                                        ["2013B", "2013J", "2014B"]]
    at.selectbox(key="presentation").set_value("2013J").run()
    by_module = [df.value for df in at.dataframe if "Flagged share" in df.value.columns][0]
    assert set(by_module["Scores"]) == {"training (in-sample)"}
    assert "This is a training presentation" in _texts(at.sidebar)


def test_training_presentation_metrics_are_labelled_in_sample(synthetic_env):
    at = _open_training(AppTest.from_file(APP, default_timeout=120).run())
    assert not at.exception
    cards = {m.label: m for m in at.metric}
    for label in ["Precision", "At-risk reached"]:
        assert cards[label].value == "Not shown"
        assert cards[label].help == "In-sample scores are optimistic and not informative."
    by_module = [df.value for df in at.dataframe if "Flagged share" in df.value.columns][0]
    assert "Recall" not in by_module.columns
    assert "shown for 2014J only" in _texts(at)


def test_capacity_slider_accepts_any_share(synthetic_env, matrix):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.slider(key="capacity").set_value(13).run()
    assert not at.exception
    n_val = int((matrix["code_presentation"] == "2014J").sum())
    assert {m.label: m.value for m in at.metric}["Flagged for contact"] == f"{round(0.13 * n_val):,}"


def _chart_specs(at) -> list[dict]:
    import json

    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


def test_tables_and_charts_are_formatted_for_reading(tables_env):
    env, prof = tables_env
    processed = env / "data" / "processed"
    dept = pd.read_csv(processed / "department_risk.csv")
    hide = (dept["code_presentation"] == "2014J") & (dept["code_module"] == "AAA")
    rate_cols = ["at_risk_rate", "share_low", "share_medium", "share_high", "rate_low", "rate_medium", "rate_high"]
    dept.loc[hide, rate_cols] = float("nan")
    dept.loc[hide, "suppressed"] = True
    dept.to_csv(processed / "department_risk.csv", index=False)
    at = _go(AppTest.from_file(APP, default_timeout=120).run(), "Modules")
    assert not at.exception
    frames = [df.value for df in at.dataframe]
    view = [f for f in frames if "High band" in f.columns][0]
    assert "Hidden (small group)" not in view.columns
    assert view.loc[view["Module"] == "AAA", "At-risk rate"].iloc[0] == "n < 20"
    profiles = [f for f in frames if "Mean clicks" in f.columns][0]
    assert {"Mean active days", "Mean days inactive before Day 30", "Mean assessments submitted",
            "Mean assessments missed"} <= set(profiles.columns)
    rates = profiles["At-risk rate"].str.rstrip("%").astype(float)
    assert rates.is_monotonic_decreasing
    assert any(e.label == "What each profile means" for e in at.expander)
    specs = _chart_specs(at)
    titles = [s["layout"].get("title", {}).get("text", "") for s in specs]
    assert all(titles), titles
    stacked = [s for s in specs if s["layout"]["title"]["text"].startswith("Share of students in each risk band")][0]
    assert "All modules" in stacked["data"][0]["x"] and stacked["data"][0]["texttemplate"] == "%{y:.0%}"
    bars = [s for s in specs if s["layout"]["title"]["text"] == "Observed at-risk rate by profile, 2014J"][0]
    assert bars["data"][0]["marker"]["color"] == "#7c6cff"
    validation = prof[prof["scope"] == "validation"].sort_values("at_risk_rate", ascending=False)
    assert bars["data"][0]["text"] == [f"{r:.0%} at risk<br>{sh:.0%} of students"
                                       for r, sh in zip(validation["at_risk_rate"], validation["share_of_scope"])]
    heights = {s["layout"].get("height") for s in specs}
    assert heights == {380}, heights


def _frame(at, column):
    return [df for df in at.dataframe if column in df.value.columns][0]


def test_funnel_table_renders_a_dash_and_right_aligned_counts(tables_env):
    import json

    at = _go(AppTest.from_file(APP, default_timeout=120).run(), "Modules")
    assert not at.exception
    frame = _frame(at, "Share of previous stage")
    shown = frame.value.reset_index(drop=True)
    assert shown.loc[0, "Stage"] == "Registered" and shown.loc[0, "Share of previous stage"] == "–"
    assert shown.loc[0, "Share of registered"] == "100%"
    assert not shown.isin(["None", "nan", "NaN"]).any().any() and shown.notna().all().all()
    assert all("," in v or len(v) <= 3 for v in shown["Students"])
    config = json.loads(frame.proto.columns)
    assert config["Students"]["alignment"] == "right"


def test_capacity_can_apply_to_each_module(tables_env, matrix):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert at.radio(key="capacity_scope").value == "the whole cohort"
    at.radio(key="capacity_scope").set_value("each module separately").run()
    assert not at.exception
    by_module = _frame(at, "Flagged share").value.set_index("Module")
    val = matrix[matrix["code_presentation"] == "2014J"]
    for module, g in val.groupby("code_module"):
        assert by_module.loc[module, "Flagged"] == f"{round(0.2 * len(g)):,}"
    assert "Highest-risk 20% of each module in 2014J, all modules." in _texts(at)


def test_calibration_chart_replaces_the_histogram(tables_env, matrix, small_final):
    from sar.service import views

    at = _go(AppTest.from_file(APP, default_timeout=120).run(), "Modules")
    assert not at.exception
    spec = [s for s in _chart_specs(at) if s["layout"]["title"]["text"].startswith("Observed at-risk rate against")][0]
    expected = views.calibration_bins(views.score_presentation(small_final[0], matrix, "2014J"))
    points = spec["data"][1]
    assert points["text"] == [f"{n:,}" for n in expected["students"]]
    assert points["y"] == pytest.approx(list(expected["observed_rate"]))
    assert spec["data"][0]["x"] == [0, 1] and spec["data"][0]["y"] == [0, 1]
    text = _texts(at)
    assert "limited number of distinct values because isotonic" in text and "Band cutoffs: Medium from" in text
    assert not [s for s in _chart_specs(at) if any(t.get("type") == "histogram" for t in s["data"])]


def test_about_explains_the_bands_design_and_limitations(tables_env, matrix, small_final):
    from sar.service import views

    env, _ = tables_env
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    about_tab = _tab(at, "About")
    about = _texts(about_tab)
    reading = [df.value for df in about_tab.dataframe if "Lift" in df.value.columns][0].set_index("Band")
    dept = pd.read_csv(env / "data" / "processed" / "department_risk.csv")
    row = dept[(dept["code_presentation"] == "2014J") & (dept["code_module"] == "All modules")].iloc[0]
    for band in ["Low", "Medium", "High"]:
        rate = row[f"rate_{band.lower()}"]
        assert reading.loc[band, "Observed at-risk rate, 2014J"] == f"{rate:.0%}"
        assert reading.loc[band, "Lift"] == f"{rate / row['at_risk_rate']:.2f}"
    assert "a Medium band is not a warning on its own" in about
    for phrase in ["Prediction day", "Day 30", "Temporal split by presentation", "Students who appear in both",
                   "out-of-fold scores", "is a validation presentation, not an untouched test set",
                   "modules (courses), not departments"]:
        assert phrase in about, phrase
    scored = views.score_presentation(small_final[0], matrix, "2014J")
    scored = scored.assign(flagged=views.flag_capacity(scored, 0.2).flagged)
    recall = views.batch_summary(scored, outcomes=True).drop(index="All modules")["Recall"].sort_values()
    lowest = "; ".join(f"{m} {r:.0%}" for m, r in recall.head(2).items())
    assert f"the lowest module recall is {lowest}" in about
    bands = [df.value for df in about_tab.dataframe if "How the cutoff was set" in df.value.columns][0]
    assert bands["How the cutoff was set"].str.contains("training data|below the Medium").all()


def test_three_d_view_caption_explains_the_encoding(empty_env):
    from conftest import FEATURES, SMALL_PARAMS, synthetic_matrix

    from sar.data.split import make_split
    from sar.models import final

    big = synthetic_matrix(n=3000, seed=3)
    processed = empty_env / "data" / "processed"
    processed.mkdir(parents=True)
    big.to_csv(processed / "day30_feature_matrix.csv", index=False)
    train, val = make_split(big)
    fitted = final.fit(train, final.final_features(FEATURES), final.forest_builder(SMALL_PARAMS))
    metrics = {"validation_presentation": ["2014J"],
               "final": {"cutoffs_from_oof": fitted["cutoffs"], "validation": final.evaluate(fitted, val)}}
    save_final(fitted, {"seed": 42}, metrics, empty_env / "models" / "final")
    at = _go(AppTest.from_file(APP, default_timeout=120).run(), "Modules")
    at.toggle(key="show_3d").set_value(True).run()
    assert not at.exception
    text = _texts(at)
    assert "days active by Day 30" in text and "average assessment score by Day 30" in text
    assert "marker size shows how many students" in text and "fewer than 20 students are not shown" in text


def test_about_works_with_a_model_but_no_data(empty_env, small_final):
    fitted, metrics = small_final
    save_final(fitted, {"seed": 42, "created_utc": "2026-01-01T00:00:00+00:00"}, metrics,
               empty_env / "models" / "final")
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    about = _about_text(at)
    assert "- Model built (UTC): 2026-01-01T00:00:00+00:00" in about and "- Day 30 data: not loaded" in about


@pytest.mark.parametrize("band", ["High", "Low"])
def test_example_buttons_choose_a_validation_student_in_the_band(synthetic_env, matrix, small_final, band):
    from sar.service import views

    at = _go(AppTest.from_file(APP, default_timeout=120).run(), "Student")
    module = at.selectbox(key="lookup_module").value
    at.button(key=f"example_{band.lower()}").click().run()
    assert not at.exception
    assert at.selectbox(key="presentation").value == "2014J"
    scored = views.score_presentation(small_final[0], matrix, "2014J")
    expected = views.example_student(scored, band, module, 42)
    assert (at.selectbox(key="lookup_module").value, at.selectbox(key="lookup_student").value) == expected
    text = _texts(at)
    assert f'">{band} risk</span>' in text and f"Module **{expected[0]}**" in text and "ranked " in text
    assert "Primary</span>" in text
    expanders = [e.label for e in _tab(at, "Student").expander]
    if band == "Low":
        assert "Low risk: support is optional." in text and "Other options" not in expanders
    else:
        assert "support is optional" not in text
    at.button(key=f"example_{band.lower()}").click().run()          # same seed, same student
    assert at.selectbox(key="lookup_student").value == expected[1] and not at.exception


def test_streamlit_config_sets_theme_and_minimal_toolbar():
    import tomllib

    config = tomllib.loads((PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["client"]["toolbarMode"] == "minimal"
    assert config["theme"]["base"] == "dark" and config["theme"]["primaryColor"] == "#7c6cff"
    assert config["theme"]["backgroundColor"] == "#0b1020" and config["theme"]["textColor"] == "#e8ecff"
    assert config["server"]["address"] == "localhost"
    source = (PROJECT_ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert 'st.set_page_config(page_title="Student Academic Risk: Day 30 early warning", layout="wide")' in source


def test_without_precomputed_explanations_only_the_top_rows_get_reasons(synthetic_env):
    at = _open_training(AppTest.from_file(APP, default_timeout=120).run(), "2013J")
    at.slider(key="capacity").set_value(50).run()
    assert not at.exception
    listing = _listing(at).value
    assert len(listing) > 50
    opened = listing["Main reason"] == "Open the student to see"
    assert not opened.iloc[:50].any() and opened.iloc[50:].all()
    assert "Reasons and suggestions are prepared for the first 50 students listed." in _texts(at)
    assert "Available for the validation presentation (2014J)." in _texts(at)


def test_navigation_renders_only_the_selected_page(synthetic_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    markers = {"Overview": "Flagged for contact", "Student": "Show an example High-risk student",
               "Modules": "### Risk bands by module", "Support": "OULAD has no intervention records",
               "About": "### Model"}
    for page, marker in markers.items():
        tab = _tab(at, page)
        text = _texts(tab) + " ".join([b.label for b in tab.button] + [m.label for m in tab.metric])
        assert not at.exception and f"## {page}" in text and "Presentation: 2014J (validation)" in text
        assert marker in text and not any(m in text for p, m in markers.items() if p != page)


def test_opening_a_listed_student_shows_the_student_page(synthetic_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    row = _listing(at).value.iloc[2]
    # The row-selection callback sets these three values; AppTest cannot click a table row.
    at.session_state["lookup_module"], at.session_state["lookup_student"] = row["Module"], int(row["Student ID"])
    at.session_state["nav_page"] = "Student"
    at.run()
    assert not at.exception and at.radio(key="nav_page").value == "Student"
    assert f"Student {row['Student ID']}" in _texts(at.main)


def test_page_selections_survive_a_visit_to_another_page(synthetic_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    module, sid = _open_student(at)
    _go(_go(at, "Modules"), "Student")
    assert (at.selectbox(key="lookup_module").value, at.selectbox(key="lookup_student").value) == (module, sid)
    assert f"Student {sid}" in _texts(at.main)


def test_review_table_filters_by_student_id_and_band(synthetic_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    full = _listing(at).value
    target = full.iloc[3]
    at.text_input(key="review_search").input(target["Student ID"]).run()
    found = _listing(at).value
    assert target["Student ID"] in set(found["Student ID"])
    assert found["Student ID"].str.contains(target["Student ID"]).all()
    assert at.session_state["priority_rows"][list(found["Student ID"]).index(target["Student ID"])] == (
        target["Module"], int(target["Student ID"]))
    at.text_input(key="review_search").input("").run()
    band = full["Risk band"].iloc[-1]
    at.selectbox(key="review_band").set_value(band).run()
    assert set(_listing(at).value["Risk band"]) == {band}
    assert list(_listing(at).value["Rank"]) == list(full.loc[full["Risk band"] == band, "Rank"])


def test_training_student_shows_the_band_without_a_ring(synthetic_env, matrix):
    at = _open_training(AppTest.from_file(APP, default_timeout=120).run(), "2013J")
    part = matrix[matrix["code_presentation"] == "2013J"].iloc[0]
    _open_student(at, part["code_module"], int(part["id_student"]))
    assert not at.exception
    text = _texts(at.main)
    assert "(in-sample)" in text and "Training presentations have in-sample scores" in text
    assert "Estimated risk" not in text and not [s for s in _chart_specs(at) if s["data"][0].get("hole") == 0.78]


def test_modules_page_charts_come_from_observed_outcomes(tables_env, matrix):
    from sar.service import views

    at = _go(AppTest.from_file(APP, default_timeout=120).run(), "Modules")
    assert not at.exception
    specs = {s["layout"]["title"]["text"]: s for s in _chart_specs(at) if s["layout"].get("title")}
    traces = {t["name"]: t for t in specs["Observed at-risk rate by intake, all modules"]["data"]}
    assert set(traces) == {"Validation", "Training"} and traces["Validation"]["x"] == ["2014J"]
    assert traces["Validation"]["marker"]["color"] == "#7c6cff" and traces["Training"]["marker"]["color"] == "#4f8cff"
    expected = views.at_risk_by_presentation(matrix).set_index("code_presentation")["at_risk_rate"]
    got = {x: y for t in traces.values() for x, y in zip(t["x"], t["y"])}
    assert sorted(got) == list(expected.index) and [got[p] for p in expected.index] == pytest.approx(list(expected))
    funnel = [s for t, s in specs.items() if t.startswith("Students at each stage")][0]["data"][0]
    assert funnel["textposition"] == "inside"
    assert set(funnel["y"]) <= {"Registered", "Still registered at Day 30", "Active in the VLE", "Submission check met",
                                "Did not withdraw", "Passed", "Submission check met (not applicable)"}
    activity = specs["Observed at-risk rate by days active by Day 30, 2014J, all modules"]["data"][0]
    buckets = views.at_risk_by_activity(matrix, "2014J")
    shown = buckets[~buckets["suppressed"]]
    assert activity["x"] == list(shown["day_band"].astype(str))
    assert activity["text"] == [f"{r:.0%}<br>n = {n:,}" for r, n in zip(shown["at_risk_rate"], shown["students"])]
    funnel = [s for t, s in specs.items() if t.startswith("Students at each stage")][0]["data"][0]
    assert len(set(funnel["marker"]["color"])) > 1                         # blue-to-violet gradient


def test_support_page_counts_the_primary_suggestion_of_each_flagged_student(synthetic_env, matrix, small_final):
    from sar.recommend.interventions import INTERVENTIONS
    from sar.service import priority, views

    models = synthetic_env / "models"
    loaded = load_final_model(models).value
    part = matrix[matrix["code_presentation"] == "2014J"]
    base, contrib, raw = explain_rows(loaded.fitted["model"], part[loaded.fitted["features"]])
    save_shap_cache(models, "2014J", part[["code_module", "code_presentation", "id_student"]], contrib, raw, base,
                    loaded.config)
    at = _go(AppTest.from_file(APP, default_timeout=120).run(), "Support")
    assert not at.exception
    supports = _listing(_go(at, "Overview")).value["Suggested support"]
    counts = priority.support_counts(supports)
    _go(at, "Support")
    cards = {m.label: m.value for m in at.main.metric}
    titles = [title for title, _ in INTERVENTIONS.values()]
    assert sorted(cards) == sorted(titles) and all(cards[t] == f"{counts[t]:,}" for t in cards)
    values = [counts[t] for t in cards]
    assert values == sorted(values, reverse=True)                             # ordered by count
    text = _texts(at.main)
    assert "rule-based" in text and "OULAD has no intervention records" in text
    assert all(definition in text for _, definition in INTERVENTIONS.values())
    for name, (title, _) in INTERVENTIONS.items():
        assert (priority.never_first(name) in text) == (counts[title] == 0)
    flagged = int(counts.sum())
    top = counts.drop(priority.NO_SUPPORT).idxmax()
    assert ("Most flagged students" in text) == (counts[top] > flagged / 2)
    specs = {s["layout"]["title"]["text"]: s for s in _chart_specs(at)}
    by_module = specs["Primary suggestion by module"]
    assert sum(sum(t["y"]) for t in by_module["data"]) == int(counts.drop(priority.NO_SUPPORT).sum())
    colours = {t["marker"]["color"] for t in by_module["data"]}
    colours |= set(specs["Primary suggestion among flagged students"]["data"][0]["marker"]["color"])
    assert not colours & {"#ff5d73", "#ffb454", "#35d0ba"}                     # never a band colour
    assert views.MIN_CELL == 20


def test_support_page_needs_prepared_suggestions(synthetic_env):
    at = _open_training(AppTest.from_file(APP, default_timeout=120).run(), "2013J")
    at.slider(key="capacity").set_value(50).run()                           # more flagged than are prepared live
    _go(at, "Support")
    assert not at.exception and "Available for the validation presentation (2014J)." in _texts(at.main)


FORBIDDEN = re.compile(r"EduGuard|last month|Intervention Outcomes|\bGPA\b|[Aa]ttendance|Assign Tutor|Mark as Assigned|"
                       r"Contact Student|Schedule Meeting|[Ff]ollow-up date|\u2014")
FORBIDDEN_COLUMNS = {"Name", "Student name", "Status", "Progress", "Follow-up", "Outcome", "GPA", "Attendance"}


def test_no_page_shows_data_the_system_does_not_have(tables_env, matrix):
    at = AppTest.from_file(APP, default_timeout=120).run()
    _open_student(at)
    seen = [_texts(at.sidebar), _texts(at.main)]
    for page in PAGES:
        tab = _tab(at, page)
        assert not at.exception
        seen.append(_texts(tab) + " ".join([b.label for b in tab.button] + [m.label for m in tab.metric]))
        for frame in tab.dataframe:
            assert not FORBIDDEN_COLUMNS & set(frame.value.columns), frame.value.columns
    text = "\n".join(seen)
    assert not FORBIDDEN.search(text), FORBIDDEN.search(text)
    assert {b.label for b in at.get("button")} <= {"Show an example High-risk student",
                                                    "Show an example Low-risk student"}


def test_sidebar_shows_each_control_only_where_it_applies(synthetic_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    expected = {"Overview": (True, True), "Student": (False, False), "Modules": (False, False),
                "Support": (True, False), "About": (False, False)}
    for page, (filters, advanced) in expected.items():
        _go(at, page)
        keys = {w.key for w in at.sidebar.selectbox} | {w.key for w in at.sidebar.slider}
        assert "presentation" in keys and ({"module_filter", "capacity"} <= keys) == filters, page
        assert any(e.label == "Advanced" for e in at.sidebar.expander) == advanced, page
    assert [o.split()[-1] for o in at.radio(key="nav_page").options] == PAGES
    assert _go(at, "Student").selectbox(key="lookup_module").label == "Module of the student"


def test_review_table_has_a_day30_snapshot_column(synthetic_env, matrix):
    from sar.service import views

    at = AppTest.from_file(APP, default_timeout=120).run()
    listing = _listing(at).value
    assert list(listing.columns) == LIST_COLUMNS
    first = listing.iloc[0]
    row = matrix[(matrix["code_presentation"] == "2014J") & (matrix["code_module"] == first["Module"])
                 & (matrix["id_student"] == int(first["Student ID"]))].iloc[0]
    days = int(row["active_days"])
    assert first["Day 30 snapshot"] == (f"{days} active day{'' if days == 1 else 's'}, {int(row['n_missed'])} missed, "
                                        f"{int(row['n_submitted'])} submitted")
    assert views.MIN_CELL == 20


def test_about_names_modules_whose_scored_work_carries_no_weight(synthetic_env, matrix):
    processed = synthetic_env / "data" / "processed"
    weightless = matrix.assign(weighted_score=matrix["weighted_score"].where(matrix["code_module"] != "BBB"))
    weightless.to_csv(processed / "day30_feature_matrix.csv", index=False)
    about = _about_text(AppTest.from_file(APP, default_timeout=120).run())
    assert "In BBB, the work scored by Day 30 carries no weight" in about
    assert "That gap is not used as a reason or a support trigger." in about
