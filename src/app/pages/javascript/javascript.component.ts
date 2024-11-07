import { Component, OnInit } from '@angular/core';
import { EnrollService } from '../services/enroll.service';

@Component({
  selector: 'ngx-javascript',
  templateUrl: './javascript.component.html',
  styleUrls: ['./javascript.component.scss'],
 // providers: [EnrollService]
})
export class JavascriptComponent implements OnInit {
  title='Javascript';

  OnEnroll(){
    //alert("You are enrolled" + this.title + " course");

    //const enrollService = new EnrollService();
    //enrollService.OnEnrollClicked(this.title);

    this.enrollService.OnEnrollClicked(this.title);
  }



  constructor(private enrollService: EnrollService) { }

  ngOnInit(): void {
  }

}
