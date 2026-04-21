import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';

import { RiskBadgeComponent } from '../../shared/risk-badge/risk-badge.component';
import { StatusBadgeComponent } from '../../shared/status-badge/status-badge.component';
import { ConfidenceBarComponent } from '../../shared/confidence-bar/confidence-bar.component';
import { SkeletonComponent } from '../../shared/skeleton/skeleton.component';
import { EmptyStateComponent } from '../../shared/empty-state/empty-state.component';
import { StudentDetailPanelComponent } from './student-detail-panel.component';

import { StudentsService, ScoredStudent } from '../../core/services/students.service';
import { RiskLevel } from '../../core/models/prediction.model';

@Component({
  selector: 'app-students-list',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    DecimalPipe,
    RiskBadgeComponent,
    StatusBadgeComponent,
    ConfidenceBarComponent,
    SkeletonComponent,
    EmptyStateComponent,
    StudentDetailPanelComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './students-list.component.html',
})
export class StudentsListComponent implements OnInit {
  protected readonly students = inject(StudentsService);
  private readonly router = inject(Router);
  private readonly route = inject(ActivatedRoute);

  readonly selectedId = signal<string | null>(null);
  readonly pageSize = signal(20);
  readonly page = signal(1);
  readonly riskLevels: (RiskLevel | 'All')[] = ['All', 'High', 'Medium', 'Low'];

  readonly filteredRows = this.students.filtered;
  readonly programmes = this.students.programmes;

  readonly pagedRows = computed(() => {
    const all = this.filteredRows();
    const start = (this.page() - 1) * this.pageSize();
    return all.slice(start, start + this.pageSize());
  });

  readonly totalPages = computed(() =>
    Math.max(1, Math.ceil(this.filteredRows().length / this.pageSize())),
  );

  readonly selected = computed<ScoredStudent | null>(() => {
    const id = this.selectedId();
    return id ? (this.students.findById(id) ?? null) : null;
  });

  ngOnInit(): void {
    this.students.load().subscribe();
    this.route.queryParamMap.subscribe((p) => {
      const id = p.get('id');
      if (id) {
        this.selectedId.set(id);
      }
    });
  }

  open(row: ScoredStudent): void {
    this.selectedId.set(row.student.student_id);
    this.router.navigate([], {
      queryParams: { id: row.student.student_id },
      queryParamsHandling: 'merge',
    });
  }

  closePanel(): void {
    this.selectedId.set(null);
    this.router.navigate([], { queryParams: { id: null }, queryParamsHandling: 'merge' });
  }

  onSearchInput(value: string): void {
    this.students.setQuery(value);
    this.page.set(1);
  }

  onRiskChange(value: string): void {
    this.students.setRisk(value as RiskLevel | 'All');
    this.page.set(1);
  }

  onProgrammeChange(value: string): void {
    this.students.setProgramme(value);
    this.page.set(1);
  }

  reset(): void {
    this.students.resetFilters();
    this.page.set(1);
  }

  prev(): void {
    this.page.update((p) => Math.max(1, p - 1));
  }

  next(): void {
    this.page.update((p) => Math.min(this.totalPages(), p + 1));
  }

  trackById(_: number, row: ScoredStudent): string {
    return row.student.student_id;
  }
}
