import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'ngx-homepage',
  templateUrl: './homepage.component.html',
  styleUrls: ['./homepage.component.scss']
})
export class HomepageComponent implements OnInit {

  constructor() { }

  ngOnInit(): void {
  }

  //string interpolation: data binding
  slogan: string = 'This is hompage slogan';

  getSlogan(){
    return 'This is function getSlogan';
  }

  //property binding
  source: string = 'assets/images/alan.png';

  display: boolean = false;

  //event binding
  

}
