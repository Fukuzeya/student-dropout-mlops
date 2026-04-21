import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { RouterLink } from '@angular/router';

import { KpiCardComponent } from '../../shared/kpi-card/kpi-card.component';
import { RiskBadgeComponent } from '../../shared/risk-badge/risk-badge.component';
import { ConfidenceBarComponent } from '../../shared/confidence-bar/confidence-bar.component';
import { SkeletonComponent } from '../../shared/skeleton/skeleton.component';
import { EmptyStateComponent } from '../../shared/empty-state/empty-state.component';

import { MonitoringService } from '../../core/services/monitoring.service';
import { StudentsService, ScoredStudent } from '../../core/services/students.service';
import { ToastService } from '../../core/services/toast.service';

import { DashboardKpis } from '../../core/models/monitoring.model';
import { RiskLevel } from '../../core/models/prediction.model';
import { num, pct, timeAgo, RISK_ORDER } from '../../core/utils/format';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    DecimalPipe,
    RouterLink,
    KpiCardComponent,
    RiskBadgeComponent,
    ConfidenceBarComponent,
    SkeletonComponent,
    EmptyStateComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './dashboard.component.html',
})
export class DashboardComponent implements OnInit {
  private readonly monitoring = inject(MonitoringService);
  protected readonly students = inject(StudentsService);
  private readonly toast = inject(ToastService);

  readonly kpis = signal<DashboardKpis | null>(null);
  readonly loadingKpis = signal(true);
  readonly loadingCohort = signal(true);

  readonly riskTotals = this.students.riskTotals;

  readonly atRiskTop = computed(() => {
    return this.students
      .all()
      .slice()
      .sort((a, b) => {
        const r = RISK_ORDER[a.prediction.risk_level] - RISK_ORDER[b.prediction.risk_level];
        if (r !== 0) return r;
        return b.prediction.probabilities.Dropout - a.prediction.probabilities.Dropout;
      })
      .filter((row) => row.prediction.risk_level !== 'Low')
      .slice(0, 8);
  });

  readonly distribution = computed(() => {
    const totals = this.riskTotals();
    const sum = totals.High + totals.Medium + totals.Low || 1;
    return {
      High: { count: totals.High, pct: (totals.High / sum) * 100 },
      Medium: { count: totals.Medium, pct: (totals.Medium / sum) * 100 },
      Low: { count: totals.Low, pct: (totals.Low / sum) * 100 },
    };
  });

  ngOnInit(): void {
    this.refresh();
  }

  refresh(): void {
    this.loadingKpis.set(true);
    this.monitoring.kpis().subscribe({
      next: (k) => {
        this.kpis.set(k);
        this.loadingKpis.set(false);
      },
      error: () => {
        this.kpis.set(null);
        this.loadingKpis.set(false);
      },
    });

    this.loadingCohort.set(true);
    this.students.load().subscribe({
      next: () => this.loadingCohort.set(false),
      error: () => this.loadingCohort.set(false),
    });
  }

  manualRefresh(): void {
    this.refresh();
    this.toast.info('Refreshing', 'Pulling latest cohort scores and KPIs.');
  }

  trackById(_: number, item: ScoredStudent): string {
    return item.student.student_id;
  }

  fmtPct = pct;
  fmtNum = num;
  fmtAgo = timeAgo;

  riskLabels: RiskLevel[] = ['High', 'Medium', 'Low'];
}
