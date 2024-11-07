/**
 * @license
 * Copyright Akveo. All Rights Reserved.
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';
import { CoreModule } from './@core/core.module';
import { ThemeModule } from './@theme/theme.module';
import { AppComponent } from './app.component';
import { AppRoutingModule } from './app-routing.module';
import {
  NbChatModule,
  NbDatepickerModule,
  NbDialogModule,
  NbMenuModule,
  NbSidebarModule,
  NbToastrModule,
  NbWindowModule,
} from '@nebular/theme';
import { HomepageComponent } from './pages/homepage/homepage.component';
import { FilterComponent } from './pages/filter/filter.component';
import { SearchComponent } from './pages/search/search.component';
import { CustomerListComponent } from './pages/customer-list/customer-list.component';
import { DemoComponent } from './pages/demo/demo.component';
import { ProductComponent } from './pages/product/product.component';
import { SetBackgroundDirective } from './pages/customdirective/set-background.directive';  
import { HighlightDirective } from './pages/customdirective/highlight.directive';  
import { FormsModule } from '@angular/forms';
import { BetterhighlightDirective } from './pages/customdirective/betterhighlight.directive';
import { ClassDirective } from './pages/customdirective/class.directive';
import { HoverDirective } from './pages/customdirective/hover.directive';
import { StyleDirective } from './pages/customdirective/style.directive';
import { IfDirective } from './pages/customdirective/if.directive';
import { JavascriptComponent } from './pages/javascript/javascript.component';
import { AngularComponent } from './pages/angular/angular.component';
import { EnrollService } from './pages/services/enroll.service';
import { AdduserComponent } from './pages/adduser/adduser.component';


@NgModule({
  declarations: [
    AppComponent,
    HomepageComponent,
    FilterComponent,
    SearchComponent,
    CustomerListComponent,
    DemoComponent,
    ProductComponent,
    SetBackgroundDirective,
    HighlightDirective,
    BetterhighlightDirective,
    HoverDirective,
    ClassDirective,
    StyleDirective,
    IfDirective,
    JavascriptComponent,
    AngularComponent,
    AdduserComponent
  ],
  imports: [
    FormsModule,
    BrowserModule,
    BrowserAnimationsModule,
    HttpClientModule,
    AppRoutingModule,
    NbSidebarModule.forRoot(),
    NbMenuModule.forRoot(),
    NbDatepickerModule.forRoot(),
    NbDialogModule.forRoot(),
    NbWindowModule.forRoot(),
    NbToastrModule.forRoot(),
    NbChatModule.forRoot({
      messageGoogleMapKey: 'AIzaSyA_wNuCzia92MAmdLRzmqitRGvCF7wCZPY',
    }),
    CoreModule.forRoot(),
    ThemeModule.forRoot(),
  ],
  providers:[EnrollService],
  bootstrap: [AppComponent],
})
export class AppModule {
}
