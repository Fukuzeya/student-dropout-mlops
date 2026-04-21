import { ChangeDetectionStrategy, Component, EventEmitter, Input, Output, computed, signal } from '@angular/core';
import { CommonModule, DatePipe, DecimalPipe } from '@angular/common';

import { RiskBadgeComponent } from '../../shared/risk-badge/risk-badge.component';
import { StatusBadgeComponent } from '../../shared/status-badge/status-badge.component';
import { ConfidenceBarComponent } from '../../shared/confidence-bar/confidence-bar.component';
import { ShapBarComponent } from '../../shared/shap-bar/shap-bar.component';

import { ScoredStudent } from '../../core/services/students.service';
import { InterventionRecommendation } from '../../core/models/prediction.model';

@Component({
  selector: 'app-student-detail-panel',
  standalone: true,
  imports: [
    CommonModule,
    DatePipe,
    DecimalPipe,
    RiskBadgeComponent,
    StatusBadgeComponent,
    ConfidenceBarComponent,
    ShapBarComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div
      class="fixed inset-0 z-40 flex justify-end animate-fade-in-up"
      role="dialog"
      aria-modal="true"
      [attr.aria-label]="'Student detail for ' + student().student_id"
    >
      <button
        type="button"
        class="absolute inset-0 bg-black/60 backdrop-blur-sm"
        aria-label="Close panel"
        (click)="closed.emit()"
      ></button>

      <aside class="relative w-full max-w-xl h-full bg-surface-900 border-l border-surface-800 shadow-card-lg overflow-y-auto">
        <header class="sticky top-0 z-10 bg-surface-900/95 backdrop-blur border-b border-surface-800 px-6 py-4 flex items-start justify-between gap-4">
          <div class="min-w-0">
            <p class="text-2xs uppercase tracking-wider text-surface-400">Student profile</p>
            <h2 class="text-lg font-semibold text-white truncate">
              {{ student().display_name ?? student().student_id }}
            </h2>
            <p class="text-xs text-surface-400 font-mono">{{ student().student_id }}</p>
          </div>
          <button type="button" class="btn-ghost" (click)="closed.emit()" aria-label="Close">×</button>
        </header>

        <div class="px-6 py-5 space-y-6">
          <section class="grid grid-cols-2 gap-3">
            <div class="surface-card p-3">
              <p class="text-2xs uppercase tracking-wider text-surface-400">Programme</p>
              <p class="text-sm text-white mt-1">{{ student().programme ?? 'Unassigned' }}</p>
            </div>
            <div class="surface-card p-3">
              <p class="text-2xs uppercase tracking-wider text-surface-400">Cohort</p>
              <p class="text-sm text-white mt-1">{{ student().cohort ?? '—' }}</p>
            </div>
            <div class="surface-card p-3">
              <p class="text-2xs uppercase tracking-wider text-surface-400">Age at enrolment</p>
              <p class="text-sm text-white mt-1">{{ student().age_at_enrollment }}</p>
            </div>
            <div class="surface-card p-3">
              <p class="text-2xs uppercase tracking-wider text-surface-400">Tuition status</p>
              <p
                class="text-sm mt-1 font-medium"
                [class.text-status-graduate]="student().tuition_fees_up_to_date === 1"
                [class.text-status-dropout]="student().tuition_fees_up_to_date !== 1"
              >
                {{ student().tuition_fees_up_to_date === 1 ? 'Up to date' : 'Outstanding' }}
              </p>
            </div>
          </section>

          <section class="surface-card p-4 space-y-4">
            <header class="flex items-center justify-between">
              <h3 class="section-title">Prediction outcome</h3>
              <p class="text-2xs text-surface-400 font-mono">
                {{ prediction().scored_at | date: 'medium' }}
              </p>
            </header>
            <div class="flex flex-wrap items-center gap-2">
              <app-status-badge [status]="prediction().predicted_class" />
              <app-risk-badge [level]="prediction().risk_level" />
              <span class="badge-neutral font-mono text-2xs">{{ prediction().model_version }}</span>
            </div>
            <app-confidence-bar [probabilities]="prediction().probabilities" />
          </section>

          <section class="surface-card p-4 space-y-3">
            <header class="flex items-center justify-between">
              <h3 class="section-title">Top SHAP contributions</h3>
              <p class="section-subtitle">Pushes prediction toward
                <span class="text-status-dropout font-medium">Dropout (right)</span> /
                <span class="text-status-graduate font-medium">Graduate (left)</span>
              </p>
            </header>
            <app-shap-bar [features]="prediction().top_shap_features" [limit]="8" />
          </section>

          <section class="surface-card p-4 space-y-3">
            <header class="flex items-center justify-between">
              <div>
                <h3 class="section-title">Recommended interventions</h3>
                <p class="section-subtitle">UZ-contextual playbook based on this risk profile</p>
              </div>
              <button type="button" class="btn-primary text-xs" (click)="trigger()">
                Trigger ({{ prediction().recommended_interventions.length }})
              </button>
            </header>

            @if (prediction().recommended_interventions.length === 0) {
              <p class="text-xs text-surface-400 italic">No interventions recommended at this risk level.</p>
            } @else {
              <ul class="space-y-2">
                @for (item of prediction().recommended_interventions; track item.code) {
                  <li class="border border-surface-700/60 rounded-lg p-3 bg-surface-850/40">
                    <div class="flex items-start justify-between gap-3">
                      <div class="min-w-0">
                        <p class="text-sm font-medium text-white">{{ item.title }}</p>
                        <p class="text-xs text-surface-400 mt-0.5">{{ item.description }}</p>
                        <p class="text-2xs text-surface-500 mt-1">
                          Owner: <span class="text-surface-200">{{ item.owner }}</span>
                          · Code: <span class="font-mono">{{ item.code }}</span>
                        </p>
                      </div>
                      <span class="badge" [ngClass]="priorityClass(item.priority)">
                        {{ item.priority | uppercase }}
                      </span>
                    </div>
                  </li>
                }
              </ul>
            }

            @if (triggered()) {
              <p class="text-2xs text-status-graduate">
                ✓ Interventions queued for the Student Welfare workflow.
              </p>
            }
          </section>
        </div>
      </aside>
    </div>
  `,
})
export class StudentDetailPanelComponent {
  @Input({ required: true }) scored!: ScoredStudent;
  @Output() closed = new EventEmitter<void>();

  readonly triggered = signal(false);
  readonly student = computed(() => this.scored.student);
  readonly prediction = computed(() => this.scored.prediction);

  trigger(): void {
    this.triggered.set(true);
  }

  priorityClass(priority: InterventionRecommendation['priority']): string {
    switch (priority) {
      case 'urgent':
        return 'badge-dropout';
      case 'high':
        return 'badge-enrolled';
      case 'medium':
        return 'badge-neutral';
      default:
        return 'badge-graduate';
    }
  }
}
