export interface ApiHealth {
  status: 'ok' | 'degraded' | 'down';
  uptime_seconds: number;
  model_version: string;
  champion_macro_f1: number;
  predictions_last_hour: number;
}

export interface DriftReportSummary {
  generated_at: string;
  drift_score: number;
  features_drifted: number;
  features_total: number;
  target_drift_detected: boolean;
  report_url: string;
}

export interface ModelRegistryEntry {
  name: string;
  version: string;
  stage: 'Production' | 'Staging' | 'Archived' | 'None';
  macro_f1: number;
  dropout_recall: number;
  registered_at: string;
}

export interface RiskDistribution {
  High: number;
  Medium: number;
  Low: number;
}

export interface DashboardKpis {
  total_students: number;
  high_risk_count: number;
  high_risk_pct: number;
  predictions_today: number;
  active_interventions: number;
  model_macro_f1: number;
  model_version: string;
  last_drift_check: string;
  drift_score: number;
}

export interface FairnessGroup {
  group: string;
  support: number;
  macro_f1: number;
  dropout_recall: number;
  dropout_flag_rate: number;
  dropout_fpr: number;
}

export interface FairnessAttribute {
  attribute: string;
  groups: FairnessGroup[];
  demographic_parity_gap: number;
  equal_opportunity_gap: number;
  predictive_equality_gap: number;
}

export interface ThresholdSweepRow {
  threshold: number;
  macro_f1: number;
  dropout_precision: number;
  dropout_recall: number;
  dropout_f1: number;
  flagged_pct: number;
}

export interface RetrainAuditEntry {
  timestamp: string;
  trigger: string;
  promoted: boolean;
  reason: string;
  champion_macro_f1: number;
  challenger_macro_f1: number;
  macro_f1_delta: number;
  per_class_deltas: Record<string, number>;
  mcnemar_p_value: number | null;
  mcnemar_b: number | null;
  mcnemar_c: number | null;
  mcnemar_significant: boolean | null;
  n_test: number;
  registered_model_name: string | null;
  registered_model_version: string | null;
}

export interface RetrainHistoryResponse {
  n: number;
  entries: RetrainAuditEntry[];
}

export interface RetrainResponse {
  promoted: boolean;
  reason: string;
  champion_macro_f1: number;
  challenger_macro_f1: number;
  per_class_deltas: Record<string, number>;
  timestamp: string;
  trigger: string;
  mcnemar_p_value: number | null;
  mcnemar_significant: boolean | null;
  n_test: number;
  registered_model_name: string | null;
  registered_model_version: string | null;
}

export interface RetrainRunStart {
  run_id: string;
  status_url: string;
  logs_url: string;
}

export type RetrainRunState = 'running' | 'succeeded' | 'failed';

export interface RetrainRunStatus {
  run_id: string;
  trigger: string;
  started_at: string;
  ended_at: string | null;
  state: RetrainRunState;
  stage: string;
  percent: number;
  log_count: number;
  logs: string[];
  result: RetrainResponse | null;
  error: string | null;
}

export interface DriftCheckResult {
  detected: boolean;
  drift_share: number;
  n_drifted: number;
  n_total: number;
  report_url: string;
}

export interface DriftAutoRetrainResponse {
  drift: DriftCheckResult;
  retrain: RetrainResponse | null;
  skipped: boolean;
  skip_reason: string | null;
  threshold: number;
}

export interface DriftRunStart {
  run_id: string;
  status_url: string;
  logs_url: string;
}

export interface DriftRunStatus {
  run_id: string;
  trigger: string;
  started_at: string;
  ended_at: string | null;
  state: RetrainRunState;
  stage: string;
  percent: number;
  log_count: number;
  logs: string[];
  drift: DriftCheckResult | null;
  retrain: RetrainResponse | null;
  skipped: boolean | null;
  skip_reason: string | null;
  threshold: number | null;
  error: string | null;
}

export interface EvaluationSummary {
  model_name: string;
  n_test: number;
  macro_f1: number;
  macro_f1_lower: number;
  macro_f1_upper: number;
  dropout_recall_argmax: number;
  dropout_recall_tuned: number;
  chosen_threshold: number;
  expected_utility_argmax: number;
  expected_utility_tuned: number;
  fairness_max_gap: number;
  fairness_summary_attribute: string;
  calibration_ece: number;
  calibration_ece_post: number | null;
  temperature: number | null;
  figure_urls: { reliability_pre?: string; reliability_post?: string };
  details: {
    classes: string[];
    headline: {
      bootstrap: Record<string, { metric: string; point: number; lower: number; upper: number }>;
      per_class: Record<string, { precision: number; recall: number; f1: number; support: number }>;
      confusion_matrix: number[][];
    };
    threshold: {
      chosen_threshold: number;
      target_recall: number;
      rationale: string;
      sweep: ThresholdSweepRow[];
    };
    fairness: {
      attributes: FairnessAttribute[];
      summary_max_gap: number;
      summary_attribute: string;
    };
    cost: {
      argmax: { total_cost: number; cost_per_sample: number; expected_utility: number };
      tuned: { total_cost: number; cost_per_sample: number; expected_utility: number };
    };
  };
}
