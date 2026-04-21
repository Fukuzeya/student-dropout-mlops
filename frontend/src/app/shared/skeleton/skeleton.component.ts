import { ChangeDetectionStrategy, Component, Input } from '@angular/core';

@Component({
  selector: 'app-skeleton',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div class="skeleton" [style.height]="height" [style.width]="width"></div>`,
})
export class SkeletonComponent {
  @Input() height = '1rem';
  @Input() width = '100%';
}
