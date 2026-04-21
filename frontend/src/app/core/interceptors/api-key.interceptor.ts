import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';

import { environment } from '../../../environments/environment';
import { AuthService } from '../services/auth.service';

const READ_PATH_PATTERNS = [
  '/predict',
  '/predict/batch',
  '/explain',
  '/students',
  '/monitoring',
  '/registry',
];

export const apiKeyInterceptor: HttpInterceptorFn = (req, next) => {
  const auth = inject(AuthService);
  const apiKey = auth.snapshot().apiKey;
  if (!apiKey) {
    return next(req);
  }
  const needsKey = READ_PATH_PATTERNS.some((p) => req.url.includes(p));
  if (!needsKey) {
    return next(req);
  }
  const cloned = req.clone({
    setHeaders: { [environment.apiKeyHeader]: apiKey },
  });
  return next(cloned);
};
