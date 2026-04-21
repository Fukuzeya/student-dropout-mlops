/**
 * Single-student payload accepted by POST /predict. Field names mirror the
 * backend Pydantic model in `backend/app/api/v1/schemas.py:StudentFeatures`
 * — the API also accepts the UCI column aliases (e.g. "Marital status") via
 * `populate_by_name=True`, but we send the Python attribute form for cleaner
 * TypeScript ergonomics.
 */
export interface StudentFeatures {
  marital_status: number;
  application_mode: number;
  application_order: number;
  course: number;
  attendance: number;
  previous_qualification: number;
  previous_qualification_grade: number;
  nacionality: number;
  mothers_qualification: number;
  fathers_qualification: number;
  mothers_occupation: number;
  fathers_occupation: number;
  admission_grade: number;
  displaced: number;
  educational_special_needs: number;
  debtor: number;
  tuition_fees_up_to_date: number;
  gender: number;
  scholarship_holder: number;
  age_at_enrollment: number;
  international: number;
  cu_1_credited: number;
  cu_1_enrolled: number;
  cu_1_evaluations: number;
  cu_1_approved: number;
  cu_1_grade: number;
  cu_1_no_eval: number;
  cu_2_credited: number;
  cu_2_enrolled: number;
  cu_2_evaluations: number;
  cu_2_approved: number;
  cu_2_grade: number;
  cu_2_no_eval: number;
  unemployment_rate: number;
  inflation_rate: number;
  gdp: number;
}

/**
 * Cohort row as returned by GET /students/scored. The backend spreads the
 * UCI feature columns into the record using the pythonic field names on
 * `StudentFeatures` (see `backend/app/api/v1/cohort.py`) so the detail
 * panel can read `student().age_at_enrollment` directly.
 */
export interface StudentRecord extends StudentFeatures {
  student_id: string;
  display_name?: string;
  programme?: string;
  cohort?: string;
}
