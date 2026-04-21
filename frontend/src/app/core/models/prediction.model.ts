export type RiskLevel = 'High' | 'Medium' | 'Low';
export type StudentStatus = 'Dropout' | 'Enrolled' | 'Graduate';

export interface ClassProbabilities {
  Dropout: number;
  Enrolled: number;
  Graduate: number;
}

export interface ShapContribution {
  feature: string;
  value: number;
  contribution: number;
}

export interface InterventionRecommendation {
  code: string;
  title: string;
  description: string;
  owner: string;
  priority: 'urgent' | 'high' | 'medium' | 'low';
}

export interface PredictionResponse {
  student_id?: string | null;
  predicted_class: StudentStatus;
  risk_level: RiskLevel;
  probabilities: ClassProbabilities;
  top_shap_features: ShapContribution[];
  recommended_interventions: InterventionRecommendation[];
  model_version: string;
  scored_at: string;
}

export interface BatchPredictionRow extends PredictionResponse {
  row_index: number;
}

export interface BatchPredictionResponse {
  total_rows: number;
  scored_rows: number;
  failed_rows: number;
  predictions: BatchPredictionRow[];
  model_version: string;
}
