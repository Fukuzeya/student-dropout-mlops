import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';

import { EvaluationSummary, FairnessAttribute } from '../../core/models/monitoring.model';
import { environment } from '../../../environments/environment';

interface FigureUrls {
  pre?: string | null;
  post?: string | null;
}

@Component({
  selector: 'app-rigor-card',
  standalone: true,
  imports: [CommonModule, DecimalPipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (data(); as e) {
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <!-- Headline + bootstrap CI -->
        <article class="surface-card p-5 lg:col-span-1">
          <p class="text-2xs uppercase tracking-wider text-surface-400">Macro-F1</p>
          <p class="mt-1 text-3xl font-semibold tabular-nums text-white">
            {{ (e.macro_f1 * 100) | number: '1.1-1' }}<span class="text-base text-surface-400">%</span>
          </p>
          <p class="text-xs text-surface-400 mt-1">
            95% bootstrap CI:
            <span class="tabular-nums text-surface-200">
              [{{ (e.macro_f1_lower * 100) | number: '1.1-1' }}%, {{ (e.macro_f1_upper * 100) | number: '1.1-1' }}%]
            </span>
          </p>

          <div class="mt-4 grid grid-cols-2 gap-3 text-sm">
            <div>
              <p class="text-2xs uppercase tracking-wider text-surface-400">Dropout recall — argmax</p>
              <p class="mt-1 text-lg font-semibold tabular-nums text-white">
                {{ (e.dropout_recall_argmax * 100) | number: '1.1-1' }}%
              </p>
            </div>
            <div>
              <p class="text-2xs uppercase tracking-wider text-surface-400">Dropout recall — tuned</p>
              <p class="mt-1 text-lg font-semibold tabular-nums" [class]="recallClass(e)">
                {{ (e.dropout_recall_tuned * 100) | number: '1.1-1' }}%
              </p>
            </div>
            <div class="col-span-2">
              <p class="text-2xs uppercase tracking-wider text-surface-400">Operating threshold</p>
              <p class="mt-1 text-lg font-semibold tabular-nums text-brand-200">
                T = {{ e.chosen_threshold | number: '1.2-2' }}
              </p>
              <p class="text-xs text-surface-400 mt-1">{{ e.details.threshold.rationale }}</p>
            </div>
          </div>
        </article>

        <!-- Calibration -->
        <article class="surface-card p-5 lg:col-span-1">
          <header class="flex items-baseline justify-between">
            <h3 class="section-title">Calibration</h3>
            @if (e.temperature) {
              <span class="badge badge-neutral">T = {{ e.temperature | number: '1.2-2' }}</span>
            }
          </header>

          <div class="mt-3 grid grid-cols-2 gap-3 text-sm">
            <div>
              <p class="text-2xs uppercase tracking-wider text-surface-400">ECE — pre</p>
              <p class="mt-1 text-lg font-semibold tabular-nums text-white">
                {{ e.calibration_ece | number: '1.3-3' }}
              </p>
            </div>
            <div>
              <p class="text-2xs uppercase tracking-wider text-surface-400">ECE — post</p>
              <p class="mt-1 text-lg font-semibold tabular-nums" [class]="eceImprovedClass(e)">
                {{ (e.calibration_ece_post ?? e.calibration_ece) | number: '1.3-3' }}
              </p>
            </div>
          </div>

          @if (figures().pre) {
            <a [href]="figures().pre" target="_blank" rel="noopener"
               class="mt-4 block overflow-hidden rounded-md border border-surface-800/80 bg-surface-950">
              <img [src]="figures().pre!" alt="Reliability pre-calibration"
                   class="w-full aspect-[6/5] object-contain" />
            </a>
          } @else {
            <p class="mt-4 text-xs text-surface-500">
              Reliability diagram unavailable — run <code>dvc repro evaluate</code>.
            </p>
          }
        </article>

        <!-- Cost -->
        <article class="surface-card p-5 lg:col-span-1">
          <h3 class="section-title">Expected cost</h3>
          <p class="section-subtitle">Lower is better. Negative utility ↔ avg intervention units.</p>

          <div class="mt-4 space-y-3">
            <div class="flex items-baseline justify-between gap-3">
              <span class="text-sm text-surface-300">Argmax baseline</span>
              <span class="tabular-nums text-white">
                {{ e.details.cost.argmax.cost_per_sample | number: '1.3-3' }} / sample
              </span>
            </div>
            <div class="flex items-baseline justify-between gap-3">
              <span class="text-sm text-surface-300">Threshold-tuned</span>
              <span class="tabular-nums" [class]="costImprovedClass(e)">
                {{ e.details.cost.tuned.cost_per_sample | number: '1.3-3' }} / sample
              </span>
            </div>
            <div class="pt-2 border-t border-surface-800/80 flex items-baseline justify-between gap-3">
              <span class="text-2xs uppercase tracking-wider text-surface-400">Δ utility</span>
              <span class="tabular-nums text-brand-200">
                {{ deltaUtility(e) | number: '1.3-3' }}
              </span>
            </div>
          </div>
        </article>
      </div>

      <!-- Fairness audit -->
      <article class="surface-card p-5 mt-4">
        <header class="flex items-baseline justify-between">
          <div>
            <h3 class="section-title">Fairness audit</h3>
            <p class="section-subtitle">
              Per-attribute parity gaps on the holdout set (post-threshold).
            </p>
          </div>
          <span class="badge" [ngClass]="fairnessSeverity(e.fairness_max_gap).cls">
            {{ fairnessSeverity(e.fairness_max_gap).label }} —
            max EO gap {{ (e.fairness_max_gap * 100) | number: '1.1-1' }}pp
          </span>
        </header>

        @if (!e.details.fairness?.attributes?.length) {
          <p class="mt-4 text-sm text-surface-400">
            No sensitive attributes available in the holdout set.
          </p>
        } @else {
          <div class="mt-4 space-y-4">
            @for (attr of e.details.fairness.attributes; track attr.attribute) {
              <details class="rounded-md border border-surface-800/80 bg-surface-900/40">
                <summary class="cursor-pointer select-none px-4 py-3 flex items-center justify-between gap-3 text-sm">
                  <span class="font-medium text-white">{{ attr.attribute }}</span>
                  <span class="flex flex-wrap gap-2 text-2xs">
                    <span class="badge badge-neutral">DP {{ (attr.demographic_parity_gap * 100) | number: '1.1-1' }}pp</span>
                    <span class="badge"
                          [ngClass]="fairnessSeverity(attr.equal_opportunity_gap).cls">
                      EO {{ (attr.equal_opportunity_gap * 100) | number: '1.1-1' }}pp
                    </span>
                    <span class="badge badge-neutral">PE {{ (attr.predictive_equality_gap * 100) | number: '1.1-1' }}pp</span>
                  </span>
                </summary>
                <div class="overflow-x-auto">
                  <table class="min-w-full text-xs">
                    <thead class="bg-surface-850/60 text-2xs uppercase tracking-wider text-surface-400">
                      <tr>
                        <th class="text-left py-2 px-4 font-medium">Group</th>
                        <th class="text-right py-2 px-4 font-medium">Support</th>
                        <th class="text-right py-2 px-4 font-medium">Macro-F1</th>
                        <th class="text-right py-2 px-4 font-medium">Dropout recall</th>
                        <th class="text-right py-2 px-4 font-medium">Flag rate</th>
                        <th class="text-right py-2 px-4 font-medium">FPR</th>
                      </tr>
                    </thead>
                    <tbody class="divide-y divide-surface-800/80">
                      @for (g of attr.groups; track g.group) {
                        <tr class="hover:bg-surface-850/60">
                          <td class="py-2 px-4 text-white">{{ g.group }}</td>
                          <td class="py-2 px-4 text-right tabular-nums">{{ g.support }}</td>
                          <td class="py-2 px-4 text-right tabular-nums">{{ (g.macro_f1 * 100) | number: '1.1-1' }}%</td>
                          <td class="py-2 px-4 text-right tabular-nums">{{ (g.dropout_recall * 100) | number: '1.1-1' }}%</td>
                          <td class="py-2 px-4 text-right tabular-nums">{{ (g.dropout_flag_rate * 100) | number: '1.1-1' }}%</td>
                          <td class="py-2 px-4 text-right tabular-nums">{{ (g.dropout_fpr * 100) | number: '1.1-1' }}%</td>
                        </tr>
                      }
                    </tbody>
                  </table>
                </div>
              </details>
            }
          </div>
        }
      </article>
    }
  `,
})
export class RigorCardComponent {
  readonly data = input.required<EvaluationSummary | null>();

  readonly figures = computed<FigureUrls>(() => {
    const d = this.data();
    if (!d) return {};
    const join = (rel?: string): string | null =>
      rel ? `${environment.apiBaseUrl}${rel.replace(/^\/api\/v1/, '')}` : null;
    return {
      pre: join(d.figure_urls.reliability_pre),
      post: join(d.figure_urls.reliability_post),
    };
  });

  fairnessSeverity(gap: number): { label: string; cls: string } {
    if (gap < 0.05) return { label: 'Equitable', cls: 'badge-graduate' };
    if (gap < 0.10) return { label: 'Watch', cls: 'badge-enrolled' };
    return { label: 'Disparate', cls: 'badge-dropout' };
  }

  recallClass(e: EvaluationSummary): string {
    return e.dropout_recall_tuned >= e.dropout_recall_argmax
      ? 'text-status-graduate'
      : 'text-status-dropout';
  }

  eceImprovedClass(e: EvaluationSummary): string {
    if (e.calibration_ece_post == null) return 'text-white';
    return e.calibration_ece_post <= e.calibration_ece
      ? 'text-status-graduate'
      : 'text-status-dropout';
  }

  costImprovedClass(e: EvaluationSummary): string {
    return e.details.cost.tuned.cost_per_sample <= e.details.cost.argmax.cost_per_sample
      ? 'text-status-graduate'
      : 'text-status-dropout';
  }

  deltaUtility(e: EvaluationSummary): number {
    return e.details.cost.tuned.expected_utility - e.details.cost.argmax.expected_utility;
  }

  // Convenience for the audit summary attribute, kept for future use.
  worstAttribute(attrs: FairnessAttribute[]): FairnessAttribute | null {
    if (!attrs?.length) return null;
    return attrs.reduce((a, b) => (a.equal_opportunity_gap >= b.equal_opportunity_gap ? a : b));
  }
}
