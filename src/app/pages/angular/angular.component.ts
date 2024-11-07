import { Component, OnInit } from '@angular/core';
import { EnrollService } from '../services/enroll.service';


@Component({
  selector: 'ngx-angular',
  templateUrl: './angular.component.html',
  styleUrls: ['./angular.component.scss'],
 // providers: [EnrollService]
})
export class AngularComponent implements OnInit {

  title: string = 'Angular';
  constructor(private enrollService: EnrollService) { }

  ngOnInit(): void {
  }

  OnEnroll(){
    //alert("You are enrolled" + this.title + " course");
      
    //const enrollService = new EnrollService();
    //enrollService.OnEnrollClicked(this.title);
    
    this.enrollService.OnEnrollClicked(this.title); 

  }
}
