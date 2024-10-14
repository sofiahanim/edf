import { Component } from '@angular/core';

@Component({
  selector: 'ngx-footer',
  styleUrls: ['./footer.component.scss'],
  template: `
    <div class="socials">
      <span class="created-by">
        Forecasting of Electricity Energy Demand (FEED) 
      </span>
    </div>
    <div class="socials">
      <a href="https://github.com/sofiahanim/edf" target="_blank" class="ion ion-social-github"></a>
    </div>
  `,
})
export class FooterComponent {
}
