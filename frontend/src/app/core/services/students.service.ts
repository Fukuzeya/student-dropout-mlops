import { Injectable, computed, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, catchError, map, of, tap } from 'rxjs';

import { environment } from '../../../environments/environment';
import { StudentRecord } from '../models/student.model';
import { PredictionResponse, RiskLevel } from '../models/prediction.model';

export interface ScoredStudent {
  student: StudentRecord;
  prediction: PredictionResponse;
}

export interface StudentFilter {
  query: string;
  risk: RiskLevel | 'All';
  programme: string | 'All';
}

@Injectable({ providedIn: 'root' })
export class StudentsService {
  private readonly http = inject(HttpClient);
  private readonly base = environment.apiBaseUrl;

  private readonly cohort = signal<ScoredStudent[]>([]);
  private readonly loading = signal(false);
  private readonly filter = signal<StudentFilter>({
    query: '',
    risk: 'All',
    programme: 'All',
  });

  readonly all = this.cohort.asReadonly();
  readonly isLoading = this.loading.asReadonly();
  readonly currentFilter = this.filter.asReadonly();

  readonly filtered = computed(() => {
    const { query, risk, programme } = this.filter();
    const q = query.trim().toLowerCase();
    return this.cohort().filter((row) => {
      if (risk !== 'All' && row.prediction.risk_level !== risk) {
        return false;
      }
      if (programme !== 'All' && row.student.programme !== programme) {
        return false;
      }
      if (!q) {
        return true;
      }
      return (
        row.student.student_id.toLowerCase().includes(q) ||
        (row.student.display_name ?? '').toLowerCase().includes(q) ||
        (row.student.programme ?? '').toLowerCase().includes(q)
      );
    });
  });

  readonly programmes = computed(() => {
    const set = new Set<string>();
    this.cohort().forEach((r) => {
      if (r.student.programme) {
        set.add(r.student.programme);
      }
    });
    return ['All', ...Array.from(set).sort()];
  });

  readonly riskTotals = computed(() => {
    const totals = { High: 0, Medium: 0, Low: 0 } as Record<RiskLevel, number>;
    this.cohort().forEach((r) => {
      totals[r.prediction.risk_level]++;
    });
    return totals;
  });

  load(): Observable<ScoredStudent[]> {
    this.loading.set(true);
    return this.http.get<ScoredStudent[]>(`${this.base}/students/scored`).pipe(
      tap((rows) => this.cohort.set(rows)),
      catchError(() => {
        // Graceful degradation — surface empty cohort if endpoint not yet available.
        this.cohort.set([]);
        return of<ScoredStudent[]>([]);
      }),
      map((rows) => {
        this.loading.set(false);
        return rows;
      }),
    );
  }

  setQuery(query: string): void {
    this.filter.update((f) => ({ ...f, query }));
  }

  setRisk(risk: RiskLevel | 'All'): void {
    this.filter.update((f) => ({ ...f, risk }));
  }

  setProgramme(programme: string | 'All'): void {
    this.filter.update((f) => ({ ...f, programme }));
  }

  resetFilters(): void {
    this.filter.set({ query: '', risk: 'All', programme: 'All' });
  }

  findById(id: string): ScoredStudent | undefined {
    return this.cohort().find((r) => r.student.student_id === id);
  }
}
