import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';

import { RiskBadgeComponent } from '../../shared/risk-badge/risk-badge.component';
import { StatusBadgeComponent } from '../../shared/status-badge/status-badge.component';
import { ConfidenceBarComponent } from '../../shared/confidence-bar/confidence-bar.component';
import { EmptyStateComponent } from '../../shared/empty-state/empty-state.component';
import { SkeletonComponent } from '../../shared/skeleton/skeleton.component';

import { PredictionService } from '../../core/services/prediction.service';
import { ToastService } from '../../core/services/toast.service';
import { BatchPredictionResponse, RiskLevel } from '../../core/models/prediction.model';

const ACCEPTED_TYPES = ['text/csv', 'application/vnd.ms-excel'];
const MAX_BYTES = 10 * 1024 * 1024;

@Component({
  selector: 'app-batch',
  standalone: true,
  imports: [
    CommonModule,
    DecimalPipe,
    RiskBadgeComponent,
    StatusBadgeComponent,
    ConfidenceBarComponent,
    EmptyStateComponent,
    SkeletonComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './batch.component.html',
})
export class BatchComponent {
  private readonly prediction = inject(PredictionService);
  private readonly toast = inject(ToastService);

  readonly file = signal<File | null>(null);
  readonly dragOver = signal(false);
  readonly result = signal<BatchPredictionResponse | null>(null);
  readonly riskFilter = signal<RiskLevel | 'All'>('All');

  readonly isUploading = this.prediction.isBatchUploading;

  readonly filteredRows = computed(() => {
    const r = this.result();
    if (!r) return [];
    const f = this.riskFilter();
    return f === 'All' ? r.predictions : r.predictions.filter((p) => p.risk_level === f);
  });

  readonly summary = computed(() => {
    const r = this.result();
    if (!r) return null;
    const totals = { High: 0, Medium: 0, Low: 0 } as Record<RiskLevel, number>;
    r.predictions.forEach((p) => totals[p.risk_level]++);
    return totals;
  });

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragOver.set(true);
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    this.dragOver.set(false);
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragOver.set(false);
    const dropped = event.dataTransfer?.files?.[0];
    if (dropped) {
      this.acceptFile(dropped);
    }
  }

  onPick(event: Event): void {
    const input = event.target as HTMLInputElement;
    const f = input.files?.[0];
    if (f) {
      this.acceptFile(f);
    }
  }

  acceptFile(f: File): void {
    if (!f.name.toLowerCase().endsWith('.csv') && !ACCEPTED_TYPES.includes(f.type)) {
      this.toast.warning('Unsupported file', 'Only CSV files are accepted.');
      return;
    }
    if (f.size > MAX_BYTES) {
      this.toast.warning('File too large', 'Maximum upload size is 10 MB.');
      return;
    }
    this.file.set(f);
    this.result.set(null);
  }

  reset(): void {
    this.file.set(null);
    this.result.set(null);
    this.riskFilter.set('All');
  }

  submit(): void {
    const f = this.file();
    if (!f) {
      return;
    }
    this.prediction.predictBatch(f).subscribe({
      next: (res) => {
        this.result.set(res);
        this.toast.success(
          'Batch scored',
          `${res.scored_rows.toLocaleString()} of ${res.total_rows.toLocaleString()} rows processed.`,
        );
      },
      error: () => {
        // error toast surfaced by error interceptor
      },
    });
  }

  setFilter(level: RiskLevel | 'All'): void {
    this.riskFilter.set(level);
  }

  downloadCsv(): void {
    const r = this.result();
    if (!r) return;
    const header = ['row_index', 'student_id', 'predicted_class', 'risk_level', 'p_dropout', 'p_enrolled', 'p_graduate'];
    const lines = [header.join(',')];
    r.predictions.forEach((p) => {
      lines.push([
        p.row_index,
        p.student_id ?? '',
        p.predicted_class,
        p.risk_level,
        p.probabilities.Dropout.toFixed(4),
        p.probabilities.Enrolled.toFixed(4),
        p.probabilities.Graduate.toFixed(4),
      ].join(','));
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `predictions-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  trackRow(_: number, row: { row_index: number }): number {
    return row.row_index;
  }

  riskFilters: (RiskLevel | 'All')[] = ['All', 'High', 'Medium', 'Low'];
}
