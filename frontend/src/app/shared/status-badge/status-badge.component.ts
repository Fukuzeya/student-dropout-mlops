import { ChangeDetectionStrategy, Component, Input } from '@angular/core';
import { NgClass } from '@angular/common';

import { StudentStatus } from '../../core/models/prediction.model';

@Component({
  selector: 'app-status-badge',
  standalone: true,
  imports: [NgClass],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <span class="badge" [ngClass]="cls()">
      <span class="h-1.5 w-1.5 rounded-full" [ngClass]="dot()"></span>
      {{ status }}
    </span>
  `,
})
export class StatusBadgeComponent {
  @Input({ required: true }) status!: StudentStatus;

  cls(): string {
    switch (this.status) {
      case 'Dropout':
        return 'badge-dropout';
      case 'Enrolled':
        return 'badge-enrolled';
      case 'Graduate':
        return 'badge-graduate';
    }
  }

  dot(): string {
    switch (this.status) {
      case 'Dropout':
        return 'bg-status-dropout';
      case 'Enrolled':
        return 'bg-status-enrolled';
      case 'Graduate':
        return 'bg-status-graduate';
    }
  }
}
