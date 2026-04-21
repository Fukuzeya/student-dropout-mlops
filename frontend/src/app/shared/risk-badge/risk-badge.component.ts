import { ChangeDetectionStrategy, Component, Input } from '@angular/core';
import { NgClass } from '@angular/common';

import { RiskLevel } from '../../core/models/prediction.model';

@Component({
  selector: 'app-risk-badge',
  standalone: true,
  imports: [NgClass],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <span class="badge" [ngClass]="badgeClass()" [attr.aria-label]="ariaLabel()">
      <span class="h-1.5 w-1.5 rounded-full" [ngClass]="dotClass()"></span>
      {{ label() }}
    </span>
  `,
})
export class RiskBadgeComponent {
  @Input({ required: true }) level!: RiskLevel;
  @Input() showLabel = true;

  label(): string {
    return this.showLabel ? `${this.level} Risk` : this.level;
  }

  ariaLabel(): string {
    return `${this.level} dropout risk`;
  }

  badgeClass(): string {
    switch (this.level) {
      case 'High':
        return 'badge-dropout';
      case 'Medium':
        return 'badge-enrolled';
      case 'Low':
        return 'badge-graduate';
    }
  }

  dotClass(): string {
    switch (this.level) {
      case 'High':
        return 'bg-status-dropout';
      case 'Medium':
        return 'bg-status-enrolled';
      case 'Low':
        return 'bg-status-graduate';
    }
  }
}
