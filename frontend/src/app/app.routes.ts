import { Routes } from '@angular/router';
import { ShellComponent } from './layout/shell/shell.component';
import { authGuard } from './core/guards/auth.guard';

export const APP_ROUTES: Routes = [
  {
    path: 'login',
    loadComponent: () =>
      import('./features/login/login.component').then((m) => m.LoginComponent),
    title: 'Sign in — UZ EWS',
  },
  {
    path: '',
    component: ShellComponent,
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'dashboard' },
      {
        path: 'dashboard',
        loadComponent: () =>
          import('./features/dashboard/dashboard.component').then((m) => m.DashboardComponent),
        title: 'Risk Intelligence — UZ EWS',
      },
      {
        path: 'students',
        loadComponent: () =>
          import('./features/students/students-list.component').then(
            (m) => m.StudentsListComponent,
          ),
        title: 'Students — UZ EWS',
      },
      {
        path: 'batch',
        loadComponent: () =>
          import('./features/batch/batch.component').then((m) => m.BatchComponent),
        title: 'Batch Predict — UZ EWS',
      },
      {
        path: 'monitoring',
        loadComponent: () =>
          import('./features/monitoring/monitoring.component').then((m) => m.MonitoringComponent),
        title: 'Monitoring — UZ EWS',
      },
      {
        path: 'admin',
        canActivate: [authGuard],
        loadComponent: () =>
          import('./features/admin/admin.component').then((m) => m.AdminComponent),
        title: 'Administration — UZ EWS',
      },
    ],
  },
  { path: '**', redirectTo: 'dashboard' },
];
