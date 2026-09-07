"""Tests for the demo hypothesis-tree API endpoints."""

import io
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def csv_file():
    """Generate a CSV file in memory."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(n)],
        "treatment": np.random.choice(["Drug", "Placebo"], n),
        "age": np.random.normal(55, 10, n).round(1),
        "blood_pressure": np.random.normal(130, 15, n).round(1),
        "sex": np.random.choice(["M", "F"], n),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


class TestCreateSession:
    def test_upload_creates_session(self, client, csv_file):
        resp = client.post(
            "/demo/hypothesis-tree/sessions",
            files={"file": ("test.csv", csv_file, "text/csv")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["profile"]["row_count"] == 60
        assert data["profile"]["column_count"] == 5
        assert data["current_question"] is not None

    def test_upload_returns_root_question(self, client, csv_file):
        resp = client.post(
            "/demo/hypothesis-tree/sessions",
            files={"file": ("test.csv", csv_file, "text/csv")},
        )
        data = resp.json()
        q = data["current_question"]
        assert q["category"] == "start_mode"
        assert len(q["options"]) == 4


class TestGetSession:
    def test_get_existing_session(self, client, csv_file):
        create_resp = client.post(
            "/demo/hypothesis-tree/sessions",
            files={"file": ("test.csv", csv_file, "text/csv")},
        )
        sid = create_resp.json()["session_id"]

        get_resp = client.get(f"/demo/hypothesis-tree/sessions/{sid}")
        assert get_resp.status_code == 200
        assert get_resp.json()["session_id"] == sid

    def test_get_invalid_session_returns_404(self, client):
        resp = client.get("/demo/hypothesis-tree/sessions/nonexistent")
        assert resp.status_code == 404

    def test_get_session_updates_active_branch(self, client, csv_file):
        create_resp = client.post(
            "/demo/hypothesis-tree/sessions",
            files={"file": ("test.csv", csv_file, "text/csv")},
        )
        data = create_resp.json()
        sid = data["session_id"]
        
        # Fork to create another branch
        fork_resp = client.post(
            f"/demo/hypothesis-tree/sessions/{sid}/forks",
            json={"node_id": data["tree"]["root_node_id"]},
        )
        new_branch_id = fork_resp.json()["new_branch_id"]
        
        # Call GET with active_branch_id parameter
        get_resp = client.get(f"/demo/hypothesis-tree/sessions/{sid}?active_branch_id={new_branch_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["tree"]["active_branch_id"] == new_branch_id

    def test_get_session_includes_analysis_result(self, client, csv_file):
        create_resp = client.post(
            "/demo/hypothesis-tree/sessions",
            files={"file": ("test.csv", csv_file, "text/csv")},
        )
        data = create_resp.json()
        sid = data["session_id"]
        branch_id = data["tree"]["active_branch_id"]
        
        # 1. Answer start_goal
        node_id = data["tree"]["root_node_id"]
        resp = client.post(f"/demo/hypothesis-tree/sessions/{sid}/answers", json={
            "parent_node_id": node_id, "option_id": "start_goal"
        })
        node_id = resp.json()["tree"]["branches"][branch_id]["node_ids"][-1]
        
        # 2. Answer goal_compare
        resp = client.post(f"/demo/hypothesis-tree/sessions/{sid}/answers", json={
            "parent_node_id": node_id, "option_id": "goal_compare"
        })
        node_id = resp.json()["tree"]["branches"][branch_id]["node_ids"][-1]
        
        # 3. Answer select_outcome -> col_blood_pressure
        resp = client.post(f"/demo/hypothesis-tree/sessions/{sid}/answers", json={
            "parent_node_id": node_id, "option_id": "col_blood_pressure"
        })
        node_id = resp.json()["tree"]["branches"][branch_id]["node_ids"][-1]
        
        # 4. Answer design_independent
        resp = client.post(f"/demo/hypothesis-tree/sessions/{sid}/answers", json={
            "parent_node_id": node_id, "option_id": "design_independent"
        })
        node_id = resp.json()["tree"]["branches"][branch_id]["node_ids"][-1]
        
        # 5. Answer select_group -> col_treatment
        resp = client.post(f"/demo/hypothesis-tree/sessions/{sid}/answers", json={
            "parent_node_id": node_id, "option_id": "col_treatment"
        })
        node_id = resp.json()["tree"]["branches"][branch_id]["node_ids"][-1]
        
        # 6. Answer group_count -> groups_two
        resp = client.post(f"/demo/hypothesis-tree/sessions/{sid}/answers", json={
            "parent_node_id": node_id, "option_id": "groups_two"
        })
        resp_data = resp.json()
        
        # Check that candidate hypotheses and analyses are returned
        assert len(resp_data["candidate_hypotheses"]) > 0
        assert len(resp_data["candidate_analyses"]) > 0
        
        hyp_id = resp_data["candidate_hypotheses"][0]["id"]
        ana_id = resp_data["candidate_analyses"][0]["id"]
        
        # Run the analysis
        run_resp = client.post(f"/demo/hypothesis-tree/sessions/{sid}/analyses", json={
            "hypothesis_id": hyp_id,
            "analysis_id": ana_id,
        })
        assert run_resp.status_code == 200
        run_data = run_resp.json()
        assert run_data["result"] is not None
        assert run_data["result"]["hypothesis_id"] == hyp_id
        
        # Now fetch session via GET and verify analysis_result is present and correct
        get_resp = client.get(f"/demo/hypothesis-tree/sessions/{sid}")
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert get_data["analysis_result"] is not None
        assert get_data["analysis_result"]["test_used"] == run_data["result"]["test_used"]
        assert get_data["analysis_result"]["hypothesis_id"] == hyp_id



class TestAnswerSubmission:
    def test_answer_advances_tree(self, client, csv_file):
        # Create session
        create_resp = client.post(
            "/demo/hypothesis-tree/sessions",
            files={"file": ("test.csv", csv_file, "text/csv")},
        )
        data = create_resp.json()
        sid = data["session_id"]
        root_node = data["tree"]["root_node_id"]

        # Submit answer
        ans_resp = client.post(
            f"/demo/hypothesis-tree/sessions/{sid}/answers",
            json={
                "parent_node_id": root_node,
                "option_id": "start_goal",
            },
        )
        assert ans_resp.status_code == 200
        ans_data = ans_resp.json()
        # Tree should have more nodes now
        assert len(ans_data["tree"]["nodes"]) > 1

    def test_answer_to_invalid_session_fails(self, client):
        resp = client.post(
            "/demo/hypothesis-tree/sessions/nonexistent/answers",
            json={"parent_node_id": "x", "option_id": "y"},
        )
        assert resp.status_code == 404


class TestFork:
    def test_fork_creates_new_branch(self, client, csv_file):
        # Create and add an answer
        create_resp = client.post(
            "/demo/hypothesis-tree/sessions",
            files={"file": ("test.csv", csv_file, "text/csv")},
        )
        data = create_resp.json()
        sid = data["session_id"]
        root_node = data["tree"]["root_node_id"]

        # Add an answer
        client.post(
            f"/demo/hypothesis-tree/sessions/{sid}/answers",
            json={"parent_node_id": root_node, "option_id": "start_goal"},
        )

        # Fork from root
        fork_resp = client.post(
            f"/demo/hypothesis-tree/sessions/{sid}/forks",
            json={"node_id": root_node},
        )
        assert fork_resp.status_code == 200
        fork_data = fork_resp.json()
        assert "new_branch_id" in fork_data
        assert len(fork_data["tree"]["branches"]) == 2


class TestDeleteSession:
    def test_delete_removes_session(self, client, csv_file):
        create_resp = client.post(
            "/demo/hypothesis-tree/sessions",
            files={"file": ("test.csv", csv_file, "text/csv")},
        )
        sid = create_resp.json()["session_id"]

        del_resp = client.delete(f"/demo/hypothesis-tree/sessions/{sid}")
        assert del_resp.status_code == 200

        # Should be gone
        get_resp = client.get(f"/demo/hypothesis-tree/sessions/{sid}")
        assert get_resp.status_code == 404

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/demo/hypothesis-tree/sessions/nonexistent")
        assert resp.status_code == 404


class TestExistingEndpointsUnchanged:
    """Verify that existing production endpoints are not broken."""

    def test_health_endpoint_works(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


class TestExecuteAnalysisRouting:
    def test_kruskal_wallis_with_two_groups_routes_to_mann_whitney(self):
        from app.routers.demo_router import _execute_analysis

        np.random.seed(42)
        df = pd.DataFrame({
            "outcome": np.concatenate([np.random.normal(10, 2, 20), np.random.normal(15, 2, 20)]),
            "group": ["A"] * 20 + ["B"] * 20,
        })
        result = _execute_analysis(
            df=df,
            test_name="kruskal_wallis",
            outcome_col="outcome",
            group_col="group",
            paired_col=None,
            predictor_col=None,
            answers=None,
        )
        assert result["test_used"] == "mann_whitney_u"

    def test_one_way_anova_with_two_groups_routes_to_ttest(self):
        from app.routers.demo_router import _execute_analysis

        np.random.seed(42)
        df = pd.DataFrame({
            "outcome": np.concatenate([np.random.normal(10, 2, 20), np.random.normal(15, 2, 20)]),
            "group": ["A"] * 20 + ["B"] * 20,
        })
        result = _execute_analysis(
            df=df,
            test_name="one_way_anova",
            outcome_col="outcome",
            group_col="group",
            paired_col=None,
            predictor_col=None,
            answers=None,
        )
        assert result["test_used"] in ("students_ttest", "welchs_ttest")
