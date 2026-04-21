import { Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, finalize, tap } from 'rxjs';

import { environment } from '../../../environments/environment';
import { StudentFeatures } from '../models/student.model';
import {
  BatchPredictionResponse,
  PredictionResponse,
} from '../models/prediction.model';

export interface PredictionHistoryEntry {
  id: number;
  studentLabel: string;
  scoredAt: string;
  result: PredictionResponse;
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private readonly http = inject(HttpClient);
  private readonly base = environment.apiBaseUrl;

  private readonly recent = signal<PredictionHistoryEntry[]>([]);
  readonly recentPredictions = this.recent.asReadonly();
  readonly isPredicting = signal(false);
  readonly isBatchUploading = signal(false);

  predictSingle(
    features: StudentFeatures,
    studentId?: string,
    label?: string,
  ): Observable<PredictionResponse> {
    this.isPredicting.set(true);
    const payload = studentId ? { student_id: studentId, features } : { features };
    return this.http.post<PredictionResponse>(`${this.base}/predict`, payload).pipe(
      tap((res) => this.recordHistory(label ?? studentId ?? 'Ad-hoc inference', res)),
      finalize(() => this.isPredicting.set(false)),
    );
  }

  predictBatch(file: File): Observable<BatchPredictionResponse> {
    this.isBatchUploading.set(true);
    const form = new FormData();
    form.append('file', file, file.name);
    return this.http
      .post<BatchPredictionResponse>(`${this.base}/predict/batch`, form)
      .pipe(finalize(() => this.isBatchUploading.set(false)));
  }

  explain(studentId: string): Observable<PredictionResponse> {
    return this.http.get<PredictionResponse>(`${this.base}/explain/${studentId}`);
  }

  clearHistory(): void {
    this.recent.set([]);
  }

  private recordHistory(label: string, result: PredictionResponse): void {
    const entry: PredictionHistoryEntry = {
      id: Date.now(),
      studentLabel: label,
      scoredAt: new Date().toISOString(),
      result,
    };
    this.recent.update((current) => [entry, ...current].slice(0, 25));
  }
}
